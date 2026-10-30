from __future__ import annotations

import asyncio
import contextlib
import json
import time
import typing
import urllib
from asyncio.tasks import Task
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, MutableMapping, NoReturn, Optional, Tuple

import sentry_sdk
import structlog
import zstandard as zstd
from pydantic import BaseModel
from redis.exceptions import RedisError
from typing_extensions import Literal, Protocol

from kodiak import (
    app_config as conf,
    queries,
)
from kodiak.events import (
    CheckRunEvent,
    PullRequestEvent,
    PullRequestReviewEvent,
    PullRequestReviewThreadEvent,
    PushEvent,
    StatusEvent,
)
from kodiak.events.status import Branch
from kodiak.pull_request import evaluate_pr, get_pr
from kodiak.queries import Client
from kodiak.redis_client import redis_bot, redis_web_api

logger = structlog.get_logger()


INGEST_QUEUE_NAMES = "kodiak_ingest_queue_names"
WEBHOOK_QUEUE_NAMES = "kodiak_webhook_queue_names"
QUEUE_PUBSUB_INGEST = "kodiak:pubsub:ingest"


def get_ingest_queue(installation_id: int) -> str:
    return f"kodiak:ingest:{installation_id}"


RETRY_RATE_SECONDS = 2
# How long the repo worker's merge marker lives without a refresh. The worker
# refreshes it every third of this while it works on a pull request, so a
# marker older than this was left by a worker that died, and nothing is
# misled by it for longer.
MERGE_TARGET_TTL_SECONDS = 180


def installation_id_from_queue(queue_name: str) -> str:
    """
    Extract the installation id from the queue names

    On restart we only have queue names, so we need to extract installation ids from the names.

    webhook:848733 -> 848733
    merge_queue:848733.chdsbd/kodiak/master -> 848733
    """
    return queue_name.partition(":")[2].partition(".")[0]


class WebhookQueueProtocol(Protocol):
    async def enqueue(self, *, event: WebhookEvent) -> None: ...

    async def enqueue_for_repo(
        self, *, event: WebhookEvent, first: bool
    ) -> int | None: ...


async def pr_event(queue: WebhookQueueProtocol, pr: PullRequestEvent) -> None:
    """
    Trigger evaluation of modified PR.
    """
    await queue.enqueue(
        event=WebhookEvent(
            repo_owner=pr.repository.owner.login,
            repo_name=pr.repository.name,
            pull_request_number=pr.number,
            target_name=pr.pull_request.base.ref,
            installation_id=str(pr.installation.id),
        )
    )


def check_run(check_run_event: CheckRunEvent) -> Iterator[WebhookEvent]:
    """
    Trigger evaluation of all PRs included in check run.
    """
    # Prevent an infinite loop when we update our check run
    if check_run_event.check_run.name == queries.CHECK_RUN_NAME:
        return
    for pr in check_run_event.check_run.pull_requests:
        # filter out pull requests for other repositories
        if pr.base.repo.id != check_run_event.repository.id:
            continue
        yield WebhookEvent(
            repo_owner=check_run_event.repository.owner.login,
            repo_name=check_run_event.repository.name,
            pull_request_number=pr.number,
            target_name=pr.base.ref,
            installation_id=str(check_run_event.installation.id),
        )


def find_branch_names_latest(sha: str, branches: list[Branch]) -> list[str]:
    """
    from the docs:
        The "branches" key is "an array of branch objects containing the status'
        SHA. Each branch contains the given SHA, but the SHA may or may not be
        the head of the branch. The array includes a maximum of 10 branches.""
    https://developer.github.com/v3/activity/events/types/#statusevent

    NOTE(chdsbd): only take branches with commit at branch head to reduce
    potential number of api requests we need to make.
    """
    return [branch.name for branch in branches if branch.commit.sha == sha]


async def status_event(queue: WebhookQueueProtocol, status_event: StatusEvent) -> None:
    """
    Trigger evaluation of all PRs associated with the status event commit SHA.
    """
    owner = status_event.repository.owner.login
    repo = status_event.repository.name
    installation_id = str(status_event.installation.id)
    log = logger.bind(owner=owner, repo=repo, install=installation_id)

    refs = find_branch_names_latest(
        sha=status_event.sha, branches=status_event.branches
    )

    async with Client(
        owner=owner, repo=repo, installation_id=installation_id
    ) as api_client:
        if len(refs) == 0:
            # when a pull request is from a fork the status event will not have
            # any `branches`, so to be able to trigger evaluation of the PR, we
            # fetch all pull requests.
            #
            # I think we could optimize this by selecting only the fork PRs, but
            # I worry that we might miss some events where `branches` is empty,
            # but not because of a fork.
            pr_results = [await api_client.get_open_pull_requests()]
            log.info("could not find refs for status_event")
        else:
            pr_requests = [
                api_client.get_open_pull_requests(head=f"{owner}:{ref}") for ref in refs
            ]
            pr_results = await asyncio.gather(*pr_requests)

        all_events: set[WebhookEvent] = set()
        for prs in pr_results:
            if prs is None:
                continue
            for pr in prs:
                all_events.add(
                    WebhookEvent(
                        repo_owner=owner,
                        repo_name=repo,
                        pull_request_number=pr.number,
                        target_name=pr.base.ref,
                        installation_id=str(installation_id),
                    )
                )
        for event in all_events:
            await queue.enqueue(event=event)


async def pr_review(
    queue: WebhookQueueProtocol,
    review: PullRequestReviewEvent | PullRequestReviewThreadEvent,
) -> None:
    """
    Trigger evaluation of the modified PR.
    """
    await queue.enqueue(
        event=WebhookEvent(
            repo_owner=review.repository.owner.login,
            repo_name=review.repository.name,
            pull_request_number=review.pull_request.number,
            target_name=review.pull_request.base.ref,
            installation_id=str(review.installation.id),
        )
    )


def get_branch_name(raw_ref: str) -> str | None:
    """
    Extract the branch name from the ref
    """
    if raw_ref.startswith("refs/heads/"):
        return raw_ref.split("refs/heads/", 1)[1]
    return None


async def push(queue: WebhookQueueProtocol, push_event: PushEvent) -> None:
    """
    Trigger evaluation of PRs that depend on the pushed branch.
    """
    owner = push_event.repository.owner.login
    repo = push_event.repository.name
    installation_id = str(push_event.installation.id)
    branch_name = get_branch_name(push_event.ref)
    log = logger.bind(ref=push_event.ref, branch_name=branch_name)
    if branch_name is None:
        log.info("could not extract branch name from ref")
        return
    async with Client(
        owner=owner, repo=repo, installation_id=installation_id
    ) as api_client:
        # find all the PRs that depend on the branch affected by this push and
        # queue them for evaluation.
        # Any PR that has a base ref matching our event ref is dependent.
        prs = await api_client.get_open_pull_requests(base=branch_name)
        if prs is None:
            log.info("api call to find pull requests failed")
            return
        for pr in prs:
            await queue.enqueue(
                event=WebhookEvent(
                    repo_owner=owner,
                    repo_name=repo,
                    pull_request_number=pr.number,
                    target_name=pr.base.ref,
                    installation_id=installation_id,
                )
            )


def compress_payload(data: dict[str, object]) -> bytes:
    cctx = zstd.ZstdCompressor()
    return cctx.compress(json.dumps(data).encode())


async def handle_webhook_event(
    queue: WebhookQueueProtocol, event_name: str, payload: dict[str, object]
) -> None:
    log = logger.bind(event_name=event_name)

    if conf.USAGE_REPORTING and event_name in conf.USAGE_REPORTING_EVENTS:
        # store events in Redis for dequeue by web api job.
        #
        # We limit the queue length to ensure that if the dequeue job fails, we
        # won't overload Redis.
        await redis_web_api.rpush(
            b"kodiak:webhook_event",
            compress_payload(dict(event_name=event_name, payload=payload)),
        )
        await redis_web_api.ltrim(
            b"kodiak:webhook_event", 0, conf.USAGE_REPORTING_QUEUE_LENGTH
        )
        log = log.bind(usage_reported=True)

    if event_name == "check_run":
        for event in check_run(CheckRunEvent.parse_obj(payload)):
            await queue.enqueue(event=event)
    elif event_name == "pull_request":
        await pr_event(queue, PullRequestEvent.parse_obj(payload))
    elif event_name == "pull_request_review":
        await pr_review(queue, PullRequestReviewEvent.parse_obj(payload))
    elif event_name == "pull_request_review_thread":
        await pr_review(queue, PullRequestReviewThreadEvent.parse_obj(payload))
    elif event_name == "push":
        await push(queue, PushEvent.parse_obj(payload))
    elif event_name == "status":
        await status_event(queue, StatusEvent.parse_obj(payload))
    else:
        log = log.bind(event_parsed=False)

    log.info("webhook_event_handled")


class WebhookEvent(BaseModel):
    repo_owner: str
    repo_name: str
    pull_request_number: int
    installation_id: str
    target_name: str

    def get_merge_queue_name(self) -> str:
        return get_merge_queue_name(self)

    def get_merge_target_queue_name(self) -> str:
        return self.get_merge_queue_name() + ":target"

    def get_webhook_queue_name(self) -> str:
        return get_webhook_queue_name(self)

    def __hash__(self) -> int:
        return (
            hash(self.repo_owner)
            + hash(self.repo_name)
            + hash(self.pull_request_number)
            + hash(self.installation_id)
        )


async def bzpopmin_with_timeout(queue_name: str) -> Tuple[bytes, bytes, float] | None:
    return await redis_bot.bzpopmin(
        [queue_name], timeout=conf.REDIS_BLOCKING_POP_TIMEOUT_SEC
    )


async def set_merge_target_marker(event: WebhookEvent, score: float) -> None:
    """
    Mark the pull request of `event` as the one its repo worker is merging
    now. Webhook evaluations read the marker to keep their statuses from
    overwriting the worker's, and to leave the head-of-queue label alone.

    The marker expires unless it is refreshed, so one left by a worker that
    died does not outlive it for long. `process_repo_queue` refreshes it
    while it works and deletes it when it is done.
    """
    target_name = event.get_merge_target_queue_name()
    async with redis_bot.pipeline(transaction=True) as pipe:
        pipe.set(target_name, event.json(), ex=MERGE_TARGET_TTL_SECONDS)
        pipe.set(target_name + ":time", str(score), ex=MERGE_TARGET_TTL_SECONDS)
        await pipe.execute()


async def process_webhook_event(
    webhook_queue: RedisWebhookQueue,
    queue_name: str,
    log: structlog.BoundLogger,
) -> None:
    log.info("block for new webhook event")
    webhook_event_json = await bzpopmin_with_timeout(queue_name)
    if webhook_event_json is None:
        return
    log.info("parsing webhook event")
    webhook_event = WebhookEvent.parse_raw(webhook_event_json[1])
    is_active_merging = (
        await redis_bot.get(webhook_event.get_merge_target_queue_name())
        == webhook_event.json().encode()
    )

    async def dequeue() -> None:
        await redis_bot.zrem(webhook_event.get_merge_queue_name(), webhook_event.json())

    async def requeue() -> None:
        await redis_bot.zadd(
            webhook_event.get_webhook_queue_name(),
            {webhook_event.json(): time.time()},
            nx=True,
        )

    async def queue_for_merge(*, first: bool) -> Optional[int]:
        return await webhook_queue.enqueue_for_repo(event=webhook_event, first=first)

    log.info("evaluate pr for webhook event")
    pr = await evaluate_pr(
        install=webhook_event.installation_id,
        owner=webhook_event.repo_owner,
        repo=webhook_event.repo_name,
        number=webhook_event.pull_request_number,
        merging=False,
        dequeue_callback=dequeue,
        requeue_callback=requeue,
        queue_for_merge_callback=queue_for_merge,
        is_active_merging=is_active_merging,
        log=log,
    )

    # The repo worker keeps `merge.head_of_queue_label` on the PR it is
    # merging. A PR that carries the label without being that PR has a stale
    # one, e.g. from a worker that died part way through a merge: take it
    # off. Decide now, after the fetch, not from `is_active_merging` above:
    # the worker sets its target marker before adding the label and clears
    # it after removing the label, so a PR fetched with the label and found
    # without the marker is never the active merge.
    if pr is not None:
        is_active_now = (
            await redis_bot.get(webhook_event.get_merge_target_queue_name())
            == webhook_event.json().encode()
        )
        if not is_active_now:
            await pr.set_head_of_queue_label(present=False)


async def webhook_event_consumer(
    *, webhook_queue: RedisWebhookQueue, queue_name: str
) -> typing.NoReturn:
    """
    Worker to process incoming webhook events from redis

    1. process mergeability information and update github check status for pr
    2. enqueue pr into repo queue for merging, if mergeability passed
    """

    # We need to define a custom Hub so that we can set the scope correctly.
    # Without creating a new hub we end up overwriting the scopes of other
    # consumers.
    #
    # https://github.com/getsentry/sentry-python/issues/147#issuecomment-432959196
    # https://github.com/getsentry/sentry-python/blob/0da369f839ee2c383659c91ea8858abcac04b869/sentry_sdk/integrations/aiohttp.py#L80-L83
    # https://github.com/getsentry/sentry-python/blob/464ca8dda09155fcc43dfbb6fa09cf00313bf5b8/sentry_sdk/integrations/asgi.py#L90-L113
    with sentry_sdk.Hub(sentry_sdk.Hub.current) as hub:
        with hub.configure_scope() as scope:
            scope.set_tag("queue", queue_name)
            scope.set_tag("installation", installation_id_from_queue(queue_name))
        log = logger.bind(
            queue=queue_name, install=installation_id_from_queue(queue_name)
        )
        log.info("start webhook event consumer")
        while True:
            await process_webhook_event(webhook_queue, queue_name, log)


async def process_repo_queue(log: structlog.BoundLogger, queue_name: str) -> None:
    log.info("block for new repo event")
    result = await bzpopmin_with_timeout(queue_name)
    if result is None:
        return
    _key, value, score = result
    webhook_event = WebhookEvent.parse_raw(value)
    target_name = webhook_event.get_merge_target_queue_name()
    await set_merge_target_marker(webhook_event, score)

    async def keep_marker_alive() -> NoReturn:
        while True:
            await asyncio.sleep(MERGE_TARGET_TTL_SECONDS / 3)
            # A failed refresh is not fatal: the marker only has to outlive
            # the gap, and the merge goes on regardless. Only cancellation,
            # from the teardown below, ends this task.
            try:
                await set_merge_target_marker(webhook_event, score)
            except RedisError:
                log.warning("failed to refresh merge marker", exc_info=True)

    async def dequeue() -> None:
        await redis_bot.zrem(webhook_event.get_merge_queue_name(), webhook_event.json())

    async def requeue() -> None:
        await redis_bot.zadd(
            webhook_event.get_webhook_queue_name(),
            {webhook_event.json(): time.time()},
            nx=True,
        )

    async def queue_for_merge(*, first: bool) -> Optional[int]:
        raise NotImplementedError

    async def set_head_of_queue_label(*, present: bool) -> None:
        """
        This PR is the head of the merge queue while we work on it. Keep
        `merge.head_of_queue_label`, if configured, on it for exactly that time.
        """
        try:
            pr = await asyncio.wait_for(
                get_pr(
                    install=webhook_event.installation_id,
                    owner=webhook_event.repo_owner,
                    repo=webhook_event.repo_name,
                    number=webhook_event.pull_request_number,
                    dequeue_callback=dequeue,
                    requeue_callback=requeue,
                    queue_for_merge_callback=queue_for_merge,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            log.warning("timeout fetching PR for head of queue label", exc_info=True)
            return
        if pr is None:
            return
        await pr.set_head_of_queue_label(present=present)

    heartbeat = asyncio.create_task(keep_marker_alive())
    try:
        await set_head_of_queue_label(present=True)
        log.info("evaluate PR for merging")
        await evaluate_pr(
            install=webhook_event.installation_id,
            owner=webhook_event.repo_owner,
            repo=webhook_event.repo_name,
            number=webhook_event.pull_request_number,
            dequeue_callback=dequeue,
            requeue_callback=requeue,
            merging=True,
            is_active_merging=False,
            queue_for_merge_callback=queue_for_merge,
            log=log,
        )
    finally:
        # Also on an exception. Until the marker goes, webhook evaluations
        # of this PR post no status and leave its head-of-queue label on.
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        try:
            await set_head_of_queue_label(present=False)
        finally:
            log.info("merge completed, remove target marker", target_name=target_name)
            await redis_bot.delete(target_name)
            await redis_bot.delete(target_name + ":time")


async def repo_queue_consumer(*, queue_name: str) -> typing.NoReturn:
    """
    Worker for a repo given by :queue_name:

    Pull webhook events off redis queue and process for mergeability.

    We only run one of these per repo as we can only merge one PR at a time
    to be efficient. This also alleviates the need of locks.
    """
    installation = installation_id_from_queue(queue_name)
    with sentry_sdk.Hub(sentry_sdk.Hub.current) as hub:
        with hub.configure_scope() as scope:
            scope.set_tag("queue", queue_name)
            scope.set_tag("installation", installation)
        log = logger.bind(queue=queue_name, install=installation)
        log.info("start repo_consumer")
        while True:
            await process_repo_queue(log, queue_name)


T = typing.TypeVar("T")


def find_position(x: typing.Iterable[T], v: T) -> typing.Optional[int]:
    for index, item in enumerate(x):
        if item == v:
            return index
    return None


ONE_DAY = int(timedelta(days=1).total_seconds())


@dataclass(frozen=True)
class TaskMeta:
    kind: Literal["repo", "webhook"]
    queue_name: str


class RedisWebhookQueue:
    def __init__(self) -> None:
        self.worker_tasks: MutableMapping[
            str, tuple[Task[NoReturn], Literal["repo", "webhook"]]
        ] = {}  # type: ignore [assignment]

    async def create(self) -> None:
        # Restart the workers. The merge queues are found by scanning: the
        # set that once listed them has not been written since the
        # per-installation sets replaced it, so a restart used to leave
        # queued pull requests waiting for the next enqueue to start their
        # worker. Nothing is being merged when we start, so a merge marker
        # found here was left by a worker that died: remove it now rather
        # than wait for it to expire.
        async for key in redis_bot.scan_iter(match="merge_queue:*"):
            queue_name = key.decode()
            if queue_name.endswith((":target", ":target:time")):
                logger.info("remove stale merge marker", key=queue_name)
                await redis_bot.delete(queue_name)
            elif await redis_bot.zcard(queue_name) > 0:
                self.start_repo_worker(queue_name=queue_name)

        for webhook_result in await redis_bot.smembers(WEBHOOK_QUEUE_NAMES):
            queue_name = webhook_result.decode()
            self.start_webhook_worker(queue_name=queue_name)

    def start_webhook_worker(self, *, queue_name: str) -> None:
        self._start_worker(
            queue_name,
            "webhook",
            webhook_event_consumer(webhook_queue=self, queue_name=queue_name),
        )

    def start_repo_worker(self, *, queue_name: str) -> None:
        self._start_worker(
            queue_name,
            "repo",
            repo_queue_consumer(
                queue_name=queue_name,
            ),
        )

    def _start_worker(
        self,
        key: str,
        kind: Literal["repo", "webhook"],
        fut: typing.Coroutine[None, None, NoReturn],
    ) -> None:
        log = logger.bind(queue_name=key, kind=kind)
        worker_task_result = self.worker_tasks.get(key)
        if worker_task_result is not None:
            worker_task, _task_kind = worker_task_result
            if not worker_task.done():
                return
            log.info("task failed")
            # task failed. record result and restart
            exception = worker_task.exception()
            log.info("exception", excep=exception)
            sentry_sdk.capture_exception(exception)
        log.info("creating task for queue")
        # create new task for queue
        self.worker_tasks[key] = (asyncio.create_task(fut), kind)

    async def enqueue(self, *, event: WebhookEvent) -> None:
        """
        add :event: to webhook queue
        """
        queue_name = get_webhook_queue_name(event)
        async with redis_bot.pipeline(transaction=True) as pipe:
            pipe.sadd(WEBHOOK_QUEUE_NAMES, queue_name)
            pipe.zadd(queue_name, {event.json(): time.time()}, nx=True)
            await pipe.execute()
        log = logger.bind(
            owner=event.repo_owner,
            repo=event.repo_name,
            number=event.pull_request_number,
            install=event.installation_id,
        )
        log.info("enqueue webhook event")
        self.start_webhook_worker(queue_name=queue_name)

    async def enqueue_for_repo(
        self, *, event: WebhookEvent, first: bool
    ) -> Optional[int]:
        """
        1. get the corresponding repo queue for event
        2. record the queue name in the installation's set, for dashboards
        3. add event, keeping its position if it is already queued
        4. start worker (will create new worker if one does not exist)

        Webhook events during a merge bring the pull request being merged
        through here too, and it re-enters the queue behind the others. The
        repo worker pops it a second time after the merge and the merging
        evaluation dequeues it as merged, without touching the head-of-queue
        label. That costs one evaluation per merge. Refusing it here, by
        reading the worker's merge marker, made the marker a lock, and one
        left by a dead worker kept its pull request out of the queue.

        returns position of event in queue
        """
        queue_name = get_merge_queue_name(event)
        async with redis_bot.pipeline(transaction=True) as pipe:
            merge_queues_by_install = f"merge_queue_by_install:{event.installation_id}"
            pipe.sadd(merge_queues_by_install, queue_name)
            pipe.expire(merge_queues_by_install, time=ONE_DAY)
            if first:
                # place at front of queue. To allow us to always place this PR at
                # the front, we should not pass only_if_not_exists.
                pipe.zadd(queue_name, {event.json(): 1.0})
            else:
                # use only_if_not_exists to prevent changing queue positions on new
                # webhook events.
                pipe.zadd(queue_name, {event.json(): time.time()}, nx=True)
            pipe.zrange(queue_name, 0, 1000, withscores=True)
            results = await pipe.execute()
        log = logger.bind(
            owner=event.repo_owner,
            repo=event.repo_name,
            number=event.pull_request_number,
            install=event.installation_id,
        )

        log.info("enqueue repo event")
        self.start_repo_worker(queue_name=queue_name)

        zrange_results = results[-1]  # type: list[tuple[bytes, float]]
        kvs = sorted(
            ((key, value) for key, value in zrange_results), key=lambda x: x[1]
        )
        return find_position((key for key, value in kvs), event.json().encode())

    def all_tasks(self) -> Iterator[tuple[TaskMeta, Task[NoReturn]]]:
        for queue_name, (task, task_kind) in self.worker_tasks.items():
            yield (TaskMeta(kind=task_kind, queue_name=queue_name), task)


def get_merge_queue_name(event: WebhookEvent) -> str:
    escaped_target = urllib.parse.quote(event.target_name)
    return f"merge_queue:{event.installation_id}.{event.repo_owner}/{event.repo_name}/{escaped_target}"


def get_webhook_queue_name(event: WebhookEvent) -> str:
    return f"webhook:{event.installation_id}"
