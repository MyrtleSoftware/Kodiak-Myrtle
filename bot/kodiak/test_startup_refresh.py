from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, cast

import httpx
import pytest
import uvicorn
from pytest_mock import MockFixture

from kodiak.config import V1
from kodiak.entrypoints.ingest import enqueue_event, refresh_when_serving
from kodiak.queries import CfgInfo
from kodiak.queue import WebhookEvent, handle_webhook_event
from kodiak.startup_refresh import (
    RefreshIncomplete,
    refresh_pull_requests,
    refresh_pull_requests_with_retry,
    should_evaluate,
)


class FakeQueue:
    def __init__(self) -> None:
        self.events: List[WebhookEvent] = []

    async def enqueue(self, *, event: WebhookEvent) -> None:
        self.events.append(event)

    async def enqueue_for_repo(self, *, event: WebhookEvent, first: bool) -> None:
        raise NotImplementedError


def pull_request(
    number: int, labels: list[str], base: str = "main", login: str = "alice"
) -> dict[str, Any]:
    return dict(
        number=number,
        base=dict(ref=base),
        labels=[dict(name=label) for label in labels],
        user=dict(login=login),
    )


def repository(
    name: str, archived: bool = False, default_branch: Optional[str] = "main"
) -> dict[str, Any]:
    return dict(
        name=name,
        full_name=f"acme/{name}",
        owner=dict(login="acme"),
        archived=archived,
        default_branch=default_branch,
    )


def config(toml: str) -> V1:
    parsed = V1.parse_toml("version = 1\n" + toml)
    assert isinstance(parsed, V1)
    return parsed


@pytest.mark.parametrize(
    "cfg, pr, expected",
    (
        (config(""), pull_request(1, ["automerge"]), True),
        (config(""), pull_request(1, ["bug"]), False),
        (config(""), pull_request(1, []), False),
        (
            config('merge.automerge_label = "ready for merge"'),
            pull_request(1, ["bug", "ready for merge"]),
            True,
        ),
        (
            config('merge.automerge_label = "ready for merge"'),
            pull_request(1, ["automerge"]),
            False,
        ),
        (
            config('merge.automerge_label = ["ship it", "ready for merge"]'),
            pull_request(1, ["ship it"]),
            True,
        ),
        (
            config("merge.require_automerge_label = false"),
            pull_request(1, []),
            True,
        ),
        (
            config('merge.automerge_dependencies.usernames = ["dependabot"]'),
            pull_request(1, [], login="dependabot[bot]"),
            True,
        ),
        (
            config('merge.automerge_dependencies.usernames = ["dependabot"]'),
            pull_request(1, [], login="alice"),
            False,
        ),
        # Without a configuration we cannot tell: any label could be it.
        (None, pull_request(1, ["bug"]), True),
        (None, pull_request(1, []), False),
    ),
)
def test_should_evaluate(cfg: V1 | None, pr: dict[str, Any], expected: bool) -> None:
    assert should_evaluate(cfg, pr) is expected


async def test_refresh_pull_requests(mocker: MockFixture, caplog: Any) -> None:
    """
    Each repository is filtered by the configuration on its default branch,
    whatever the base branch of the pull request, in every installation and
    across pages. Archived repositories are skipped, and a
    failing installation does not stop the others, but makes the refresh
    raise at the end so that it is tried again.
    """
    responses = {
        "/app/installations": (
            [dict(id=1), dict(id=2)],
            "/app/installations?page=2",
        ),
        "/app/installations?page=2": ([dict(id=3)], None),
        "/installation/repositories": (
            dict(
                repositories=[
                    repository("widgets"),
                    repository("quiet"),
                    repository("bare", default_branch="develop"),
                    # Only the organization's configuration is read, with
                    # a warning.
                    repository("branchless", default_branch=None),
                    repository("old", True),
                ]
            ),
            None,
        ),
        "/repos/acme/widgets/pulls": (
            [
                pull_request(1, ["ready for merge"]),
                pull_request(2, ["automerge"]),
                pull_request(3, []),
            ],
            "/repos/acme/widgets/pulls?page=2",
        ),
        "/repos/acme/widgets/pulls?page=2": (
            [
                pull_request(4, ["bug", "ready for merge"]),
                pull_request(5, ["ready for merge"], base="release"),
                pull_request(6, ["bug"], base="release"),
            ],
            None,
        ),
        # No configuration: any label could be the automerge label.
        "/repos/acme/bare/pulls": (
            [pull_request(7, ["bug"]), pull_request(8, [])],
            None,
        ),
        "/repos/acme/branchless/pulls": (
            [pull_request(9, ["automerge"]), pull_request(10, ["bug"])],
            None,
        ),
        "/repos/acme/quiet/pulls": ([], None),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] == "token install-2":
            return httpx.Response(500)
        page = request.url.params.get("page")
        key = request.url.path + (f"?page={page}" if page else "")
        body, next_path = responses[key]
        headers = {}
        if next_path is not None:
            headers["Link"] = f'<https://api.github.com{next_path}>; rel="next"'
        return httpx.Response(200, json=body, headers=headers)

    mocker.patch(
        "kodiak.startup_refresh.HttpClient",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    mocker.patch("kodiak.startup_refresh.generate_jwt", return_value="jwt")
    throttled: List[str] = []

    class FakeThrottler:
        def __init__(self, installation_id: str) -> None:
            self.installation_id = installation_id

        async def __aenter__(self) -> None:
            throttled.append(self.installation_id)

        async def __aexit__(self, *args: object) -> None:
            pass

    mocker.patch(
        "kodiak.startup_refresh.get_thottler_for_installation",
        lambda *, installation_id: FakeThrottler(installation_id),
    )

    async def fake_get_headers(*, session: Any, installation_id: str) -> Any:
        return dict(Authorization=f"token install-{installation_id}")

    mocker.patch("kodiak.startup_refresh.get_headers", fake_get_headers)

    configs: Dict[Optional[str], CfgInfo] = {
        "main": CfgInfo(
            parsed=config('merge.automerge_label = "ready for merge"'),
            text="",
            file_expression="",
        ),
        # the configuration in the organization's `.github` repository
        None: CfgInfo(parsed=config(""), text="", file_expression=""),
    }
    config_requests: List[tuple[str, Optional[str], Optional[str]]] = []

    class FakeClient:
        def __init__(self, *, owner: str, repo: str, installation_id: str) -> None:
            self.repo = repo

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def send_query(self, **kwargs: Any) -> Dict[str, Any]:
            return dict(
                data=dict(orgConfigRepo=dict(defaultBranchRef=dict(name="trunk")))
            )

        async def get_config_for_ref(
            self, *, ref: str | None, org_repo_default_branch: str | None
        ) -> CfgInfo | None:
            config_requests.append((self.repo, ref, org_repo_default_branch))
            return configs.get(ref)

    mocker.patch("kodiak.startup_refresh.Client", FakeClient)

    enqueued: List[Tuple[int, str, Dict[str, Any]]] = []

    async def enqueue(
        installation_id: int, event_name: str, payload: Dict[str, Any]
    ) -> None:
        enqueued.append((installation_id, event_name, payload))

    with pytest.raises(RefreshIncomplete):
        await refresh_pull_requests(enqueue)

    assert [installation_id for installation_id, _, _ in enqueued] == [1] * 5 + [3] * 5
    assert (
        len(
            [
                msg
                for _mod, _level, msg in caplog.record_tuples
                if "could not find default branch" in msg and "branchless" in msg
            ]
        )
        == 2
    )
    # The worker handles the events as it handles webhooks from GitHub.
    queue = FakeQueue()
    for _, event_name, payload in enqueued:
        await handle_webhook_event(queue, event_name, payload)

    assert [
        (e.installation_id, e.pull_request_number, e.target_name) for e in queue.events
    ] == [
        ("1", 1, "main"),
        ("1", 4, "main"),
        ("1", 5, "release"),
        ("1", 7, "main"),
        ("1", 9, "main"),
        ("3", 1, "main"),
        ("3", 4, "main"),
        ("3", 5, "release"),
        ("3", 7, "main"),
        ("3", 9, "main"),
    ]
    assert queue.events[0] == WebhookEvent(
        repo_owner="acme",
        repo_name="widgets",
        pull_request_number=1,
        installation_id="1",
        target_name="main",
    )
    # Every REST request goes through the throttler of its installation.
    assert throttled == ["kodiak"] * 2 + ["1"] * 6 + ["2"] + ["3"] * 6
    # One configuration read per repository, from its default branch, and
    # none for a repository without open pull requests.
    assert (
        config_requests
        == [
            ("widgets", "main", "trunk"),
            ("bare", "develop", "trunk"),
            ("branchless", None, "trunk"),
        ]
        * 2
    )


async def test_refresh_pull_requests_with_retry(mocker: MockFixture) -> None:
    """
    A failed scan is tried again until it completes, and abandoned after
    REFRESH_MAX_ATTEMPTS.
    """
    mocker.patch("kodiak.startup_refresh.REFRESH_RETRY_DELAY_SECONDS", 0)
    mocker.patch("kodiak.startup_refresh.REFRESH_MAX_ATTEMPTS", 4)
    calls = 0
    failures = 2

    async def fake_refresh(enqueue: Any) -> None:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise RefreshIncomplete

    mocker.patch("kodiak.startup_refresh.refresh_pull_requests", fake_refresh)

    async def enqueue(
        installation_id: int, event_name: str, payload: Dict[str, Any]
    ) -> None:
        raise NotImplementedError

    await refresh_pull_requests_with_retry(enqueue)
    assert calls == 3

    calls = 0
    failures = 100
    await refresh_pull_requests_with_retry(enqueue)
    assert calls == 4


async def test_refresh_when_serving(mocker: MockFixture) -> None:
    """
    Ingest starts the scan only after uvicorn has bound its socket, and the
    scan queues its events as ingest queues webhooks.
    """
    enqueues: List[Any] = []

    async def fake_refresh(enqueue: Any) -> None:
        enqueues.append(enqueue)

    mocker.patch(
        "kodiak.entrypoints.ingest.refresh_pull_requests_with_retry", fake_refresh
    )

    class FakeServer:
        started = False

    server = FakeServer()
    task = asyncio.create_task(refresh_when_serving(cast(uvicorn.Server, server)))
    await asyncio.sleep(0.3)
    assert not enqueues
    server.started = True
    await asyncio.wait_for(task, timeout=1)
    assert enqueues == [enqueue_event]
