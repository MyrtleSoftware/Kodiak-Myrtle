from typing import List

import pytest
import structlog
from pytest_mock import MockFixture

from kodiak.queue import (
    RedisWebhookQueue,
    WebhookEvent,
    installation_id_from_queue,
    process_repo_queue,
    process_webhook_event,
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
    after it, and the target marker brackets the same interval.
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
