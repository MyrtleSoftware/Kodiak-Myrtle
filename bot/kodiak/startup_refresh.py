"""
Queue pull requests for evaluation after the webhook listener starts.

GitHub does not resend webhooks that we missed. If someone adds the automerge
label while the ingest process is down, the pull request waits until some
other event arrives for it. To prevent this, ingest lists the open pull
requests of every installation when it starts, and queues a `pull_request`
webhook event for each one that Kodiak could merge, through the same path as
a webhook from GitHub.

The scan starts after ingest accepts webhooks. A label added before the scan
reads the pull request is found by the scan, and a label added after ingest
starts arrives as a webhook. If the scan started first, a label added between
the scan and the start of ingest would be lost.

To keep the number of evaluations small, we read each repository's
configuration and queue only the pull requests that the evaluation would act
on: those with one of the automerge labels, all of them when
`merge.require_automerge_label` is disabled, and those by authors in
`merge.automerge_dependencies.usernames`. If we cannot read a configuration,
we queue every pull request with a label, and the evaluation decides.
"""

from __future__ import annotations

import asyncio
import collections
from typing import Any, AsyncIterator, Awaitable, Callable, Dict

import structlog

from kodiak import app_config as conf
from kodiak.config import V1
from kodiak.http import HttpClient, HTTPError
from kodiak.queries import (
    APPLICATION_ID,
    Client,
    generate_jwt,
    get_headers,
    get_org_config_default_branch,
)
from kodiak.throttle import get_thottler_for_installation

logger = structlog.get_logger()

REFRESH_MAX_ATTEMPTS = 5
REFRESH_RETRY_DELAY_SECONDS = 60

GET_ORG_CONFIG_DEFAULT_BRANCH_QUERY = """
query ($owner: String!) {
  orgConfigRepo: repository(owner: $owner, name: ".github") {
    defaultBranchRef {
      name
    }
  }
}
"""


# Called with the installation id, the event name and the event payload.
EnqueueEvent = Callable[[int, str, Dict[str, Any]], Awaitable[None]]


class RefreshIncomplete(Exception):
    pass


def _api_headers(headers: dict[str, str]) -> dict[str, str]:
    if (
        conf.GITHUB_API_HEADER_NAME is not None
        and conf.GITHUB_API_HEADER_VALUE is not None
    ):
        headers = {**headers, conf.GITHUB_API_HEADER_NAME: conf.GITHUB_API_HEADER_VALUE}
    return headers


async def _get_pages(
    http: HttpClient, url: str, headers: dict[str, str], *, installation_id: str
) -> AsyncIterator[Any]:
    """
    Yield the JSON body of each page, following the "next" links. The
    requests are throttled like the other requests for the installation.
    """
    throttler = get_thottler_for_installation(installation_id=installation_id)
    next_url: str | None = url
    while next_url is not None:
        async with throttler:
            res = await http.get(next_url, headers=_api_headers(headers))
        res.raise_for_status()
        yield res.json()
        next_url = res.links.get("next", {}).get("url")


def should_evaluate(config: V1 | None, pull_request: dict[str, Any]) -> bool:
    """
    Whether the evaluation could act on this pull request, by the same rules
    as `kodiak.evaluation.mergeable`. It may say yes when the evaluation does
    nothing, but it must not say no when the evaluation would act.
    """
    labels = {label["name"] for label in pull_request["labels"]}
    if config is None:
        return bool(labels)
    merge = config.merge
    if not merge.require_automerge_label:
        return True
    automerge_labels = (
        {merge.automerge_label}
        if isinstance(merge.automerge_label, str)
        else set(merge.automerge_label)
    )
    if labels & automerge_labels:
        return True
    # GraphQL, which the evaluation reads, gives bot logins without the
    # "[bot]" suffix that the REST API adds.
    login: str = pull_request["user"]["login"]
    if login.endswith("[bot]"):
        login = login[: -len("[bot]")]
    return login in merge.automerge_dependencies.usernames


async def get_installation_ids(http: HttpClient) -> list[str]:
    app_token = generate_jwt(
        private_key=conf.PRIVATE_KEY, app_identifier=conf.GITHUB_APP_ID
    )
    headers = dict(
        Accept="application/vnd.github+json",
        Authorization=f"Bearer {app_token}",
    )
    installation_ids: list[str] = []
    async for page in _get_pages(
        http,
        conf.v3_url("/app/installations?per_page=100"),
        headers,
        # not a real installation: the throttler for requests as the app
        installation_id=APPLICATION_ID,
    ):
        installation_ids.extend(str(installation["id"]) for installation in page)
    return installation_ids


async def get_repository_events(
    http: HttpClient,
    headers: dict[str, str],
    installation_id: str,
    repository: dict[str, Any],
    org_default_branches: dict[str, str | None],
) -> list[dict[str, Any]]:
    """
    The payloads of `pull_request` webhook events, with only the fields that
    `kodiak.events.PullRequestEvent` reads.
    """
    owner: str = repository["owner"]["login"]
    name: str = repository["name"]
    pull_requests_by_base: dict[str, list[Any]] = collections.defaultdict(list)
    async for page in _get_pages(
        http,
        conf.v3_url(f"/repos/{owner}/{name}/pulls?state=open&per_page=100"),
        headers,
        installation_id=installation_id,
    ):
        for pull_request in page:
            pull_requests_by_base[pull_request["base"]["ref"]].append(pull_request)
    if not pull_requests_by_base:
        return []

    events: list[dict[str, Any]] = []
    async with Client(owner=owner, repo=name, installation_id=installation_id) as api:
        if owner not in org_default_branches:
            res = await api.send_query(
                query=GET_ORG_CONFIG_DEFAULT_BRANCH_QUERY,
                variables=dict(owner=owner),
                installation_id=installation_id,
            )
            org_default_branches[owner] = get_org_config_default_branch(
                (res or {}).get("data") or {}
            )
        for base, pull_requests in pull_requests_by_base.items():
            # The evaluation reads the configuration from the base branch.
            # None means that there is no configuration or that we could not
            # read it. We cannot tell which, so we let the evaluation decide.
            cfg = await api.get_config_for_ref(
                ref=base, org_repo_default_branch=org_default_branches[owner]
            )
            config = (
                cfg.parsed if cfg is not None and isinstance(cfg.parsed, V1) else None
            )
            events.extend(
                dict(
                    installation=dict(id=int(installation_id)),
                    number=pull_request["number"],
                    pull_request=dict(base=dict(ref=base)),
                    repository=dict(name=name, owner=dict(login=owner)),
                )
                for pull_request in pull_requests
                if should_evaluate(config, pull_request)
            )
    return events


async def refresh_pull_requests(enqueue: EnqueueEvent) -> None:
    """
    Queue a webhook event for each open pull request that the evaluation
    could act on, in every installation. A failure in one installation or
    repository does not stop the others, but raises RefreshIncomplete at the
    end, so that the caller can try again.
    """
    log = logger.bind(task="refresh_pull_requests")
    failed = False
    async with HttpClient() as http:
        for installation_id in await get_installation_ids(http):
            install_log = log.bind(install=installation_id)
            try:
                headers = await get_headers(
                    session=http, installation_id=installation_id
                )
                repositories: list[Any] = []
                async for page in _get_pages(
                    http,
                    conf.v3_url("/installation/repositories?per_page=100"),
                    headers,
                    installation_id=installation_id,
                ):
                    repositories.extend(
                        repository
                        for repository in page["repositories"]
                        if not repository["archived"]
                    )
            except HTTPError:
                install_log.exception("failed to list repositories")
                failed = True
                continue

            org_default_branches: dict[str, str | None] = {}
            count = 0
            for repository in repositories:
                try:
                    events = await get_repository_events(
                        http,
                        headers,
                        installation_id,
                        repository,
                        org_default_branches,
                    )
                except HTTPError:
                    install_log.exception(
                        "failed to list pull requests", repo=repository["full_name"]
                    )
                    failed = True
                    continue
                for event in events:
                    await enqueue(int(installation_id), "pull_request", event)
                count += len(events)
            install_log.info("pull requests queued", count=count)
    if failed:
        raise RefreshIncomplete


async def refresh_pull_requests_with_retry(enqueue: EnqueueEvent) -> None:
    """
    A scan that fails is tried again a few times. Each try scans everything:
    the worker ignores an event for a pull request that is already queued.
    """
    log = logger.bind(task="refresh_pull_requests")
    for attempt in range(1, REFRESH_MAX_ATTEMPTS + 1):
        try:
            await refresh_pull_requests(enqueue)
        except Exception:
            log.exception("refresh failed", attempt=attempt)
        else:
            return
        if attempt < REFRESH_MAX_ATTEMPTS:
            await asyncio.sleep(REFRESH_RETRY_DELAY_SECONDS * attempt)
    log.error("refresh abandoned")
