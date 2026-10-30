import asyncio
from typing import List

import pytest
import structlog
from pytest_mock import MockFixture
from redis.exceptions import ConnectionError as RedisConnectionError

from kodiak.queue import (
    MERGE_TARGET_TTL_SECONDS,
    RedisWebhookQueue,
    WebhookEvent,
    installation_id_from_queue,
    process_repo_queue,
    process_webhook_event,
    set_merge_target_marker,
)
from kodiak.redis_client import create_connection
from kodiak.tests.fixtures import requires_redis


@pytest.mark.parametrize(
    "queue_name, expected_installation_id",
    (
        ("merge_queue:11256551.sbdchd/squawk/main.test.foo", "11256551"),
        ("merge_queue:11256551.sbdchd/squawk", "11256551"),
        ("merge_queue:11256551.sbdchd/squawk:repo/main:test.branch", "11256551"),
        ("webhook:11256551", "11256551"),
        ("", ""),
    ),
)
def test_installation_id_from_queue(
    queue_name: str, expected_installation_id: str
) -> None:
    """
    We should gracefully parse an installation id from the queue name
    """
    assert installation_id_from_queue(queue_name) == expected_installation_id


@requires_redis
async def test_process_repo_queue_head_of_queue_label(mocker: MockFixture) -> None:
    """
    While process_repo_queue works on a pull request it is the head of the
    merge queue: the label is set before the merging evaluation and cleared
    after it, and the target marker brackets the same interval. The marker
    expires, so a worker that dies does not leave it behind for long.
    """
    # A client of our own, bound to this test's event loop, in place of the
    # module-level one created at import time.
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)

    event = WebhookEvent(
        repo_owner="acme",
        repo_name="widgets",
        pull_request_number=7,
        installation_id="4321",
        target_name="main",
    )
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name, target_name + ":time")
    await redis.zadd(queue_name, {event.json(): 1.0})

    calls: List[object] = []

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            calls.append(present)

    async def fake_get_pr(**kwargs: object) -> FakePR:
        return FakePR()

    async def fake_evaluate_pr(**kwargs: object) -> None:
        assert kwargs["merging"] is True
        assert await redis.get(target_name) == event.json().encode()
        assert 0 < await redis.ttl(target_name) <= MERGE_TARGET_TTL_SECONDS
        assert 0 < await redis.ttl(target_name + ":time") <= MERGE_TARGET_TTL_SECONDS
        calls.append("evaluate")

    mocker.patch("kodiak.queue.get_pr", fake_get_pr)
    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    await process_repo_queue(structlog.get_logger(), queue_name)

    assert calls == [True, "evaluate", False]
    assert await redis.get(target_name) is None
    await redis.connection_pool.disconnect()


@requires_redis
@pytest.mark.parametrize("active", [True, False])
async def test_process_webhook_event_head_of_queue_label(
    mocker: MockFixture, active: bool
) -> None:
    """
    After a webhook evaluation the label is cleared only if the PR is not the
    active merge at that moment. The marker is read after the evaluation, so
    a PR the repo worker took as head during our fetch keeps its label.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)

    event = WebhookEvent(
        repo_owner="acme",
        repo_name="widgets",
        pull_request_number=8,
        installation_id="4321",
        target_name="main",
    )
    queue_name = event.get_webhook_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name)
    await redis.zadd(queue_name, {event.json(): 1.0})

    calls: List[object] = []

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            calls.append(present)

    async def fake_evaluate_pr(**kwargs: object) -> FakePR:
        assert kwargs["merging"] is False
        # the repo worker takes this PR as head while we evaluate it
        if active:
            await redis.set(target_name, event.json())
        return FakePR()

    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    await process_webhook_event(RedisWebhookQueue(), queue_name, structlog.get_logger())

    assert calls == ([] if active else [False])
    await redis.delete(target_name)
    await redis.connection_pool.disconnect()


def make_event(number: int) -> WebhookEvent:
    return WebhookEvent(
        repo_owner="acme",
        repo_name="widgets",
        pull_request_number=number,
        installation_id="4321",
        target_name="main",
    )


@requires_redis
async def test_enqueue_for_repo_during_merge(mocker: MockFixture) -> None:
    """
    A webhook event during a merge puts the PR being merged back into the
    merge queue, as upstream does. The merge marker is not a lock: the repo
    worker pops the PR a second time after the merge and dequeues it as
    merged. Refusing the enqueue instead left a PR out of the queue for as
    long as a dead worker's marker survived.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    mocker.patch.object(RedisWebhookQueue, "start_repo_worker")
    event = make_event(9)
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name, target_name + ":time")
    queue = RedisWebhookQueue()

    await set_merge_target_marker(event, 1.0)
    assert await queue.enqueue_for_repo(event=event, first=False) == 0
    assert await redis.zscore(queue_name, event.json()) is not None, "re-enqueued"

    await redis.delete(queue_name, target_name, target_name + ":time")
    await redis.connection_pool.disconnect()


@requires_redis
async def test_process_repo_queue_refreshes_marker(mocker: MockFixture) -> None:
    """
    The marker outlives its expiry only while the worker refreshes it. Here
    the evaluation takes longer than the expiry and the marker is still there
    at its end, and gone once the worker is done.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    mocker.patch("kodiak.queue.MERGE_TARGET_TTL_SECONDS", 1)
    event = make_event(12)
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name, target_name + ":time")
    await redis.zadd(queue_name, {event.json(): 1.0})

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            pass

    async def fake_get_pr(**kwargs: object) -> FakePR:
        return FakePR()

    async def fake_evaluate_pr(**kwargs: object) -> None:
        await asyncio.sleep(1.6)
        assert await redis.get(target_name) == event.json().encode()
        assert await redis.get(target_name + ":time") == b"1.0"
        assert await redis.ttl(target_name) > 0

    mocker.patch("kodiak.queue.get_pr", fake_get_pr)
    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    await process_repo_queue(structlog.get_logger(), queue_name)

    assert await redis.get(target_name) is None
    assert await redis.get(target_name + ":time") is None
    await redis.connection_pool.disconnect()


@requires_redis
async def test_process_repo_queue_clears_marker_and_label_on_error(
    mocker: MockFixture,
) -> None:
    """
    If the merging evaluation raises, the marker and the label still go.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    event = make_event(10)
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name, target_name + ":time")
    await redis.zadd(queue_name, {event.json(): 1.0})

    calls: List[object] = []

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            calls.append(present)

    async def fake_get_pr(**kwargs: object) -> FakePR:
        return FakePR()

    async def fake_evaluate_pr(**kwargs: object) -> None:
        raise RuntimeError("boom")

    mocker.patch("kodiak.queue.get_pr", fake_get_pr)
    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    with pytest.raises(RuntimeError):
        await process_repo_queue(structlog.get_logger(), queue_name)

    assert calls == [True, False]
    assert await redis.get(target_name) is None
    await redis.connection_pool.disconnect()


@requires_redis
async def test_process_repo_queue_survives_failed_refresh(mocker: MockFixture) -> None:
    """
    A refresh that fails, e.g. Redis away for a moment, neither ends the
    merge nor skips the teardown. The next refresh keeps the marker alive.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    mocker.patch("kodiak.queue.MERGE_TARGET_TTL_SECONDS", 1)
    event = make_event(14)
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name, target_name, target_name + ":time")
    await redis.zadd(queue_name, {event.json(): 1.0})

    real_set_marker = set_merge_target_marker
    refreshes = 0

    async def flaky_set_marker(event: WebhookEvent, score: float) -> None:
        nonlocal refreshes
        refreshes += 1
        if refreshes == 2:
            raise RedisConnectionError("redis away")
        await real_set_marker(event, score)

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            pass

    async def fake_get_pr(**kwargs: object) -> FakePR:
        return FakePR()

    async def fake_evaluate_pr(**kwargs: object) -> None:
        await asyncio.sleep(1.6)
        assert await redis.get(target_name) == event.json().encode()

    mocker.patch("kodiak.queue.set_merge_target_marker", flaky_set_marker)
    mocker.patch("kodiak.queue.get_pr", fake_get_pr)
    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    await process_repo_queue(structlog.get_logger(), queue_name)

    assert refreshes >= 3
    assert await redis.get(target_name) is None
    await redis.connection_pool.disconnect()


@requires_redis
async def test_create_removes_stale_markers(mocker: MockFixture) -> None:
    """
    At startup nothing is being merged, so a marker a dead worker left is
    removed. It is found by scanning for it, with no bookkeeping to be kept
    in step: the only thing written here is what the worker itself writes.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    start_repo_worker = mocker.patch.object(RedisWebhookQueue, "start_repo_worker")
    mocker.patch.object(RedisWebhookQueue, "start_webhook_worker")
    event = make_event(11)
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis.delete(queue_name)
    await set_merge_target_marker(event, 1.0)

    await RedisWebhookQueue().create()

    assert await redis.get(target_name) is None
    assert await redis.get(target_name + ":time") is None
    started = [c.kwargs["queue_name"] for c in start_repo_worker.call_args_list]
    assert queue_name not in started, "an empty queue gets its worker on enqueue"
    await redis.connection_pool.disconnect()


@requires_redis
async def test_create_starts_workers_for_queues_with_entries(
    mocker: MockFixture,
) -> None:
    """
    Pull requests queued when the worker stopped are picked up again at
    startup, not at the next enqueue to their queue.
    """
    redis = create_connection()
    mocker.patch("kodiak.queue.redis_bot", redis)
    mocker.patch.object(RedisWebhookQueue, "start_repo_worker")
    mocker.patch.object(RedisWebhookQueue, "start_webhook_worker")
    event = make_event(13)
    queue_name = event.get_merge_queue_name()
    await redis.delete(queue_name)
    assert await RedisWebhookQueue().enqueue_for_repo(event=event, first=False) == 0

    start_repo_worker = mocker.patch.object(RedisWebhookQueue, "start_repo_worker")
    await RedisWebhookQueue().create()

    started = [c.kwargs["queue_name"] for c in start_repo_worker.call_args_list]
    assert queue_name in started
    await redis.delete(queue_name)
    await redis.connection_pool.disconnect()
