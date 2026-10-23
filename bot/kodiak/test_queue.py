from typing import List

import pytest
import structlog
from pytest_mock import MockFixture

from kodiak.queue import WebhookEvent, installation_id_from_queue, process_repo_queue
from kodiak.redis_client import redis_bot
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
    event = WebhookEvent(
        repo_owner="acme",
        repo_name="widgets",
        pull_request_number=7,
        installation_id="4321",
        target_name="main",
    )
    queue_name = event.get_merge_queue_name()
    target_name = event.get_merge_target_queue_name()
    await redis_bot.delete(queue_name, target_name, target_name + ":time")
    await redis_bot.zadd(queue_name, {event.json(): 1.0})

    calls: List[object] = []

    class FakePR:
        async def set_head_of_queue_label(self, *, present: bool) -> None:
            calls.append(present)

    async def fake_get_pr(**kwargs: object) -> FakePR:
        return FakePR()

    async def fake_evaluate_pr(**kwargs: object) -> None:
        assert kwargs["merging"] is True
        assert await redis_bot.get(target_name) == event.json().encode()
        calls.append("evaluate")

    mocker.patch("kodiak.queue.get_pr", fake_get_pr)
    mocker.patch("kodiak.queue.evaluate_pr", fake_evaluate_pr)

    await process_repo_queue(structlog.get_logger(), queue_name)

    assert calls == [True, "evaluate", False]
    assert await redis_bot.get(target_name) is None
