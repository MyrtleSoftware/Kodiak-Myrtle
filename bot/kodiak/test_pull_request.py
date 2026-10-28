from __future__ import annotations

import json
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    cast,
)

import pytest
from typing_extensions import Protocol

import kodiak.http as requests
from kodiak import pull_request
from kodiak.config import V1, Merge, MergeMethod, UpdateMethod
from kodiak.errors import ApiCallException, GitHubApiInternalServerError, MergeFailed
from kodiak.http import Request
from kodiak.pull_request import PRV2, EventInfoResponse, QueueForMergeCallback
from kodiak.queries import (
    BranchProtectionRule,
    Client,
    MergeableState,
    MergeStateStatus,
    NodeListPushAllowance,
    PullRequest,
    PullRequestAuthor,
    PullRequestBranchUpdateMethod,
    PullRequestState,
    RepoInfo,
    ReviewThreadConnection,
)


def create_event() -> EventInfoResponse:
    config = V1(
        version=1, merge=Merge(automerge_label="automerge", method=MergeMethod.squash)
    )
    pr = PullRequest(
        id="e14ff7599399478fb9dbc2dacb87da72",
        number=100,
        author=PullRequestAuthor(login="arnold", databaseId=49118, type="Bot"),
        mergeStateStatus=MergeStateStatus.BEHIND,
        state=PullRequestState.OPEN,
        isDraft=False,
        mergeable=MergeableState.MERGEABLE,
        isCrossRepository=False,
        labels=["automerge"],
        latest_sha="8d728d017cac4f5ba37533debe65730abe65730a",
        baseRefName="master",
        headRefName="df825f90-9825-424c-a97e-733522027e4c",
        title="Update README.md",
        body="",
        bodyText="",
        bodyHTML="",
        url="https://github.com/delos-corp/hive-mind/pull/324",
        reviewThreads=ReviewThreadConnection(nodes=[]),
    )
    rep_info = RepoInfo(
        merge_commit_allowed=False,
        rebase_merge_allowed=False,
        squash_merge_allowed=True,
        is_private=True,
        delete_branch_on_merge=False,
    )
    branch_protection = BranchProtectionRule(
        requiresStatusChecks=True,
        requiredStatusCheckContexts=[
            "ci/circleci: frontend_lint",
            "ci/circleci: frontend_test",
        ],
        requiresStrictStatusChecks=True,
        requiresCommitSignatures=False,
        requiresConversationResolution=False,
        restrictsPushes=False,
        pushAllowances=NodeListPushAllowance(nodes=[]),
    )

    return EventInfoResponse(
        config=config,
        config_str="""\
version = 1
[merge]
method = "squash"
""",
        config_file_expression="master:.kodiak.toml",
        head_exists=True,
        pull_request=pr,
        repository=rep_info,
        branch_protection=branch_protection,
        ruleset_rules=[],
        review_requests=[],
        bot_reviews=[],
        status_contexts=[],
        check_runs=[],
        valid_merge_methods=[MergeMethod.squash],
        subscription=None,
    )


async def noop(*args: object, **kwargs: object) -> None:
    return None


class BaseMockFunc:
    calls: List[Mapping[str, Any]]

    def __init__(self) -> None:
        self.calls = []

    def log_call(self, args: Dict[str, Any]) -> None:
        self.calls.append(args)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def called(self) -> bool:
        return self.call_count > 0

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: id={id(self)} call_count={self.call_count!r} called={self.called!r} calls={self.calls!r}>"


class MockMergePullRequest(BaseMockFunc):
    response: requests.Response

    async def __call__(
        self,
        number: int,
        merge_method: str,
        commit_title: str,
        commit_message: str,
        sha: str,
    ) -> requests.Response:
        self.log_call(
            dict(
                number=number,
                merge_method=merge_method,
                commit_title=commit_title,
                commit_message=commit_message,
                sha=sha,
            )
        )
        return self.response


class MockGetMergeResult(BaseMockFunc):
    responses: List[requests.Response]

    async def __call__(self, number: int, uuid: str) -> requests.Response:
        self.log_call(dict(number=number, uuid=uuid))
        return self.responses[self.call_count - 1]


class MockDeleteLabel(BaseMockFunc):
    response: requests.Response

    async def __call__(self, label: str, pull_number: int) -> requests.Response:
        self.log_call(dict(label=label, pull_number=pull_number))
        return self.response


class MockAddLabel(BaseMockFunc):
    response: requests.Response

    async def __call__(self, label: str, pull_number: int) -> requests.Response:
        self.log_call(dict(label=label, pull_number=pull_number))
        return self.response


class MockUpdateBranch(BaseMockFunc):
    response: requests.Response

    async def __call__(self, pull_number: int) -> requests.Response:
        self.log_call(dict(pull_number=pull_number))
        return self.response


class MockEnqueuePullRequest(BaseMockFunc):
    response: Optional[Dict[str, Any]]

    async def __call__(
        self, *, pull_request_id: str, expected_head_oid: str
    ) -> Optional[Dict[str, Any]]:
        self.log_call(
            dict(pull_request_id=pull_request_id, expected_head_oid=expected_head_oid)
        )
        return self.response


class MockUpdatePullRequestBranch(BaseMockFunc):
    response: Optional[Dict[str, Any]]

    async def __call__(
        self, *, pull_request_id: str, update_method: PullRequestBranchUpdateMethod
    ) -> Optional[Dict[str, Any]]:
        self.log_call(
            dict(pull_request_id=pull_request_id, update_method=update_method)
        )
        return self.response


class MockUpdateRef(BaseMockFunc):
    response: requests.Response

    async def __call__(self, *, ref: str, sha: str) -> requests.Response:
        self.log_call(dict(ref=ref, sha=sha))
        return self.response


class FakeClientProtocol(Protocol):
    merge_pull_request: MockMergePullRequest
    get_merge_result: MockGetMergeResult
    enqueue_pull_request: MockEnqueuePullRequest
    delete_label: MockDeleteLabel
    add_label: MockAddLabel
    update_branch: MockUpdateBranch
    update_pull_request_branch: MockUpdatePullRequestBranch
    update_ref: MockUpdateRef

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    async def __aenter__(self) -> FakeClientProtocol: ...

    async def __aexit__(
        self, exc_type: object, exc_value: object, traceback: object
    ) -> None: ...


def create_client() -> Type[FakeClientProtocol]:
    class FakeClient:
        merge_pull_request = MockMergePullRequest()
        get_merge_result = MockGetMergeResult()
        enqueue_pull_request = MockEnqueuePullRequest()
        delete_label = MockDeleteLabel()
        add_label = MockAddLabel()
        update_branch = MockUpdateBranch()
        update_pull_request_branch = MockUpdatePullRequestBranch()
        update_ref = MockUpdateRef()

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(
            self, exc_type: object, exc_value: object, traceback: object
        ) -> None:
            pass

    return FakeClient


def create_response(content: bytes, status_code: int) -> requests.Response:
    return requests.Response(
        status_code=status_code, content=content, request=Request(method="", url="")
    )


def create_prv2(
    event: Optional[EventInfoResponse] = None,
    install: str = "88443234",
    owner: str = "delos",
    repo: str = "incite",
    number: int = 8634,
    dequeue_callback: Callable[[], Awaitable[None]] = noop,
    queue_for_merge_callback: QueueForMergeCallback = noop,
    requeue_callback: Callable[[], Awaitable[None]] = noop,
    client: Optional[Type[FakeClientProtocol]] = None,
) -> PRV2:
    return PRV2(
        event=event if event is not None else create_event(),
        install=install,
        owner=owner,
        repo=repo,
        number=number,
        dequeue_callback=dequeue_callback,
        queue_for_merge_callback=queue_for_merge_callback,
        requeue_callback=requeue_callback,
        client=cast(Type[Client], client if client is not None else create_client()),
    )


MERGE_UUID = "630b9d5e-3f2a-4f7e-8b0c-2d5f9a8c1e42"


def merge_result_body(status: str, message: str = "", uuid: str = MERGE_UUID) -> bytes:
    """
    A merge-async body in the shape GitHub's OpenAPI description shows for
    the 200, 202, 400 and 409 responses and for the poll endpoint.
    """
    details: Dict[str, Any] = {"message": message}
    if status == "pending":
        details.update(
            uuid=uuid,
            merge_method="squash",
            merge_action="direct_merge",
            expected_head_sha="8d728d017cac4f5ba37533debe65730abe65730a",
        )
    if status == "merged":
        details["sha"] = "6dcb09b5b57875f334f61aebed695e2e4193db5e"
    return json.dumps(dict(status=status, details=details)).encode()


@pytest.fixture(autouse=True)
def no_merge_poll_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pull_request, "MERGE_POLL_RATE_SECONDS", 0)


async def test_pr_v2_merge() -> None:
    """
    We should start an async merge pinned to the head we evaluated and poll it
    until it has merged.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("pending", "Merge request enqueued."),
        status_code=202,
    )
    client.get_merge_result.responses = [
        create_response(
            content=merge_result_body("pending", "Merge request is in progress."),
            status_code=200,
        ),
        create_response(
            content=merge_result_body("merged", "Pull request was merged."),
            status_code=200,
        ),
    ]

    pr_v2 = create_prv2(client=client)
    await pr_v2.merge("squash", commit_title="my title", commit_message="my message")
    assert client.merge_pull_request.call_count == 1
    assert (
        client.merge_pull_request.calls[0]["sha"] == pr_v2.event.pull_request.latest_sha
    )
    assert client.get_merge_result.call_count == 2
    assert client.get_merge_result.calls[0] == dict(
        number=pr_v2.number, uuid=MERGE_UUID
    )


async def test_pr_v2_merge_already_merged() -> None:
    """
    A 200 means the pull request has already merged, so there is nothing to
    poll.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("merged", "Pull request is already merged."),
        status_code=200,
    )
    client.get_merge_result.responses = []

    pr_v2 = create_prv2(client=client)
    await pr_v2.merge("squash", commit_title="my title", commit_message="my message")
    assert client.get_merge_result.call_count == 0


async def test_pr_v2_merge_already_running() -> None:
    """
    A 409 names the merge that is already running, likely one we started
    before a retry. We should wait on it instead of starting another.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body(
            "pending",
            "A merge request already exists for this pull request.",
            uuid="a1b2c3",
        ),
        status_code=409,
    )
    client.get_merge_result.responses = [
        create_response(content=merge_result_body("merged"), status_code=200),
    ]

    pr_v2 = create_prv2(client=client)
    await pr_v2.merge("squash", commit_title="my title", commit_message="my message")
    assert client.merge_pull_request.call_count == 1
    assert client.get_merge_result.calls[0]["uuid"] == "a1b2c3"


async def test_pr_v2_merge_not_ready() -> None:
    """
    A 400 carries a failed result with GitHub's reason. Nothing was merged and
    there is nothing to poll.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("failed", "Pull request is closed."),
        status_code=400,
    )
    client.get_merge_result.responses = []

    pr_v2 = create_prv2(client=client)
    with pytest.raises(MergeFailed) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.message == "Pull request is closed."
    assert client.get_merge_result.call_count == 0


async def test_pr_v2_merge_failed() -> None:
    """
    A merge that fails in the background is reported with GitHub's reason.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("pending"), status_code=202
    )
    client.get_merge_result.responses = [
        create_response(
            content=merge_result_body(
                "failed", "Merge conflict: the pull request could not be merged."
            ),
            status_code=200,
        ),
    ]

    pr_v2 = create_prv2(client=client)
    with pytest.raises(MergeFailed) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.message == "Merge conflict: the pull request could not be merged."


async def test_pr_v2_merge_failed_without_message() -> None:
    """
    A failed result without a message still raises MergeFailed.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b'{"status": "failed"}', status_code=400
    )

    pr_v2 = create_prv2(client=client)
    with pytest.raises(MergeFailed) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.message == "GitHub gave no reason"


async def test_pr_v2_merge_enqueued() -> None:
    """
    We merge directly, so an enqueued result is unexpected. We should raise
    ApiCallException to retry; the re-evaluation sees the merge queue.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("pending"), status_code=202
    )
    client.get_merge_result.responses = [
        create_response(
            content=merge_result_body(
                "enqueued", "Pull request is in the merge queue."
            ),
            status_code=200,
        ),
    ]

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.method == "pull_request/merge_result"
    assert b"enqueued" in e.value.response


async def test_pr_v2_merge_poll_server_error() -> None:
    """
    A 500 while polling is safe to retry, unlike a 500 when starting the merge.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("pending"), status_code=202
    )
    client.get_merge_result.responses = [
        create_response(content=b"<html>Server Error</html>", status_code=500),
    ]

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.method == "pull_request/merge_result"
    assert e.value.status_code == 500


async def test_pr_v2_merge_start_server_error() -> None:
    """
    A 500 when starting the merge is not safe to retry.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b"<html>Server Error</html>", status_code=500
    )

    pr_v2 = create_prv2(client=client)
    with pytest.raises(GitHubApiInternalServerError):
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )


async def test_pr_v2_merge_poll_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    We should stop waiting on a merge that does not finish in time and raise
    ApiCallException, so the retry resumes waiting via the 409.
    """
    monkeypatch.setattr(pull_request, "MERGE_POLL_TIMEOUT_SECONDS", -1)
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=merge_result_body("pending"), status_code=202
    )
    client.get_merge_result.responses = []

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.method == "pull_request/merge"
    assert e.value.status_code == 202
    assert client.get_merge_result.call_count == 0


async def test_pr_v2_merge_pending_without_uuid() -> None:
    """
    A pending result we cannot poll raises ApiCallException to retry.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b'{"status": "pending", "details": {"message": "Merge request enqueued."}}',
        status_code=202,
    )
    client.get_merge_result.responses = []

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.status_code == 202
    assert client.get_merge_result.call_count == 0


async def test_pr_v2_merge_unparsable_result() -> None:
    """
    A 2xx response whose body is not a merge result raises ApiCallException.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b"<html>Accepted</html>", status_code=202
    )

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert e.value.method == "pull_request/merge"
    assert e.value.status_code == 202


async def test_pr_v2_merge_rebase_error() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b"""{"message":"This branch can't be rebased","documentation_url":"https://developer.github.com/v3/pulls/#merge-a-pull-request-merge-button"}""",
        status_code=405,
    )

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert client.merge_pull_request.call_count == 1
    assert e.value.method == "pull_request/merge"
    assert e.value.status_code == 405
    assert b"merge-a-pull-request-merge-button" in e.value.response


async def test_pr_v2_merge_service_unavailable() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.merge_pull_request.response = create_response(
        content=b"""<html>Service Unavailable</html>""", status_code=503
    )

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.merge(
            "squash", commit_title="my title", commit_message="my message"
        )
    assert client.merge_pull_request.call_count == 1
    assert e.value.method == "pull_request/merge"
    assert e.value.status_code == 503
    assert b"Service Unavailable" in e.value.response


async def test_pr_v2_add_to_merge_queue() -> None:
    """
    We should be able to add a pull request to GitHub's merge queue.
    """
    client = create_client()
    client.enqueue_pull_request.response = {
        "data": {"enqueuePullRequest": {"clientMutationId": None}}
    }

    pr_v2 = create_prv2(client=client)
    await pr_v2.add_to_merge_queue()
    assert client.enqueue_pull_request.call_count == 1
    assert client.enqueue_pull_request.calls[0] == dict(
        pull_request_id=pr_v2.event.pull_request.id,
        expected_head_oid=pr_v2.event.pull_request.latest_sha,
    )


async def test_pr_v2_add_to_merge_queue_graphql_error() -> None:
    """
    We should raise ApiCallException when the GraphQL API returns errors.
    """
    client = create_client()
    client.enqueue_pull_request.response = {
        "data": None,
        "errors": [{"message": "Pull request is in unmergeable state"}],
    }

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.add_to_merge_queue()
    assert client.enqueue_pull_request.call_count == 1
    assert e.value.method == "pull_request/enqueue"
    assert b"unmergeable state" in e.value.response


async def test_pr_v2_add_to_merge_queue_request_error() -> None:
    """
    We should raise ApiCallException when we fail to reach the GitHub API.
    """
    client = create_client()
    client.enqueue_pull_request.response = None

    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.add_to_merge_queue()
    assert client.enqueue_pull_request.call_count == 1
    assert e.value.method == "pull_request/enqueue"


async def test_pr_v2_update_branch_ok() -> None:
    """
    We should be able to update a branch.
    """
    client = create_client()
    client.update_branch.response = create_response(content=b"", status_code=204)
    pr_v2 = create_prv2(client=client)
    await pr_v2.update_branch(update_method=UpdateMethod.merge)
    assert client.update_branch.call_count == 1
    assert client.update_branch.calls[0]["pull_number"] == pr_v2.number
    assert client.update_pull_request_branch.called is False


async def test_pr_v2_update_branch_service_unavailable() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.update_branch.response = create_response(
        content=b"<html>Service Unavailable</html>", status_code=503
    )
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.update_branch(update_method=UpdateMethod.merge)
    assert client.update_branch.call_count == 1
    assert client.update_branch.calls[0]["pull_number"] == pr_v2.number
    assert e.value.method == "pull_request/update_branch"
    assert e.value.status_code == 503
    assert b"Service Unavailable" in e.value.response


async def test_pr_v2_update_branch_rebase_ok() -> None:
    """
    We should update a branch via the GraphQL API when the rebase update
    method is configured.
    """
    client = create_client()
    client.update_pull_request_branch.response = {
        "data": {"updatePullRequestBranch": {"clientMutationId": None}}
    }
    pr_v2 = create_prv2(client=client)
    await pr_v2.update_branch(update_method=UpdateMethod.rebase)
    assert client.update_branch.called is False
    assert client.update_pull_request_branch.call_count == 1
    assert client.update_pull_request_branch.calls[0] == dict(
        pull_request_id=pr_v2.event.pull_request.id,
        update_method=PullRequestBranchUpdateMethod.REBASE,
    )


async def test_pr_v2_update_branch_rebase_graphql_error() -> None:
    """
    We should raise ApiCallException when the GraphQL API returns errors for a
    rebase update.
    """
    client = create_client()
    client.update_pull_request_branch.response = {
        "data": None,
        "errors": [{"message": "Pull request is not behind the base branch"}],
    }
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.update_branch(update_method=UpdateMethod.rebase)
    assert client.update_branch.called is False
    assert client.update_pull_request_branch.call_count == 1
    assert e.value.method == "pull_request/update_branch"
    assert b"not behind the base branch" in e.value.response


async def test_pr_v2_update_branch_rebase_request_error() -> None:
    """
    We should raise ApiCallException when we fail to reach the GitHub API for a
    rebase update.
    """
    client = create_client()
    client.update_pull_request_branch.response = None
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.update_branch(update_method=UpdateMethod.rebase)
    assert client.update_branch.called is False
    assert client.update_pull_request_branch.call_count == 1
    assert e.value.method == "pull_request/update_branch"


async def test_pr_v2_add_label_ok() -> None:
    """
    We should be able to add a label.
    """
    client = create_client()
    client.add_label.response = create_response(content=b"", status_code=204)
    pr_v2 = create_prv2(client=client)
    await pr_v2.add_label(label="some-label-to-delete")
    assert client.add_label.call_count == 1
    assert client.add_label.calls[0]["label"] == "some-label-to-delete"


async def test_pr_v2_add_label_service_unavailable() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.add_label.response = create_response(
        content=b"<html>Service Unavailable</html>", status_code=503
    )
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.add_label(label="some-label-to-delete")
    assert client.add_label.call_count == 1
    assert client.add_label.calls[0]["label"] == "some-label-to-delete"
    assert e.value.method == "pull_request/add_label"
    assert e.value.status_code == 503
    assert b"Service Unavailable" in e.value.response


async def test_pr_v2_remove_label_ok() -> None:
    """
    Check that remove_label works when
    """
    client = create_client()
    client.delete_label.response = create_response(content=b"", status_code=204)
    pr_v2 = create_prv2(client=client)
    await pr_v2.remove_label(label="some-label-to-delete")
    assert client.delete_label.call_count == 1
    assert client.delete_label.calls[0]["label"] == "some-label-to-delete"


async def test_pr_v2_remove_label_service_unavailable() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.delete_label.response = create_response(
        content=b"<html>Service Unavailable</html>", status_code=503
    )
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.remove_label(label="some-label-to-delete")
    assert client.delete_label.call_count == 1
    assert client.delete_label.calls[0]["label"] == "some-label-to-delete"
    assert e.value.method == "pull_request/delete_label"
    assert e.value.status_code == 503
    assert b"Service Unavailable" in e.value.response


async def test_update_ref_ok() -> None:
    """
    Check that update_ref works
    """
    client = create_client()
    client.update_ref.response = create_response(content=b"", status_code=200)
    pr_v2 = create_prv2(client=client)
    await pr_v2.update_ref(ref="master", sha="aa218f56b14c9653891f9e74264a383fa43fefbd")
    assert client.update_ref.call_count == 1
    assert client.update_ref.calls[0] == dict(
        ref="master", sha="aa218f56b14c9653891f9e74264a383fa43fefbd"
    )


async def test_update_ref_service_unavailable() -> None:
    """
    We should raise ApiCallException when we get a bad API response.
    """
    client = create_client()
    client.update_ref.response = create_response(
        content=b"<html>Service Unavailable</html>", status_code=503
    )
    pr_v2 = create_prv2(client=client)
    with pytest.raises(ApiCallException) as e:
        await pr_v2.update_ref(
            ref="master", sha="aa218f56b14c9653891f9e74264a383fa43fefbd"
        )
    assert client.update_ref.call_count == 1
    assert client.update_ref.calls[0] == dict(
        ref="master", sha="aa218f56b14c9653891f9e74264a383fa43fefbd"
    )
    assert e.value.method == "pull_request/update_ref"
    assert e.value.status_code == 503
    assert b"Service Unavailable" in e.value.response


async def test_pr_v2_set_head_of_queue_label_adds_when_missing() -> None:
    """
    With merge.head_of_queue_label configured, present=True adds the label
    when the pull request does not have it.
    """
    client = create_client()
    client.add_label.response = create_response(content=b"", status_code=204)
    event = create_event()
    assert isinstance(event.config, V1)
    event.config.merge.head_of_queue_label = "merge-queue-head"
    event.pull_request.labels = ["automerge"]
    pr_v2 = create_prv2(event=event, client=client)
    await pr_v2.set_head_of_queue_label(present=True)
    assert client.add_label.call_count == 1
    assert client.add_label.calls[0]["label"] == "merge-queue-head"
    assert client.delete_label.call_count == 0


async def test_pr_v2_set_head_of_queue_label_removes_when_present() -> None:
    """
    present=False removes the label when the pull request has it, and leaves
    the pull request alone when it does not.
    """
    client = create_client()
    client.delete_label.response = create_response(content=b"", status_code=204)
    event = create_event()
    assert isinstance(event.config, V1)
    event.config.merge.head_of_queue_label = "merge-queue-head"
    event.pull_request.labels = ["automerge", "merge-queue-head"]
    pr_v2 = create_prv2(event=event, client=client)
    await pr_v2.set_head_of_queue_label(present=False)
    assert client.delete_label.call_count == 1
    assert client.delete_label.calls[0]["label"] == "merge-queue-head"

    event.pull_request.labels = ["automerge"]
    await pr_v2.set_head_of_queue_label(present=False)
    assert client.delete_label.call_count == 1, "nothing to remove"
    assert client.add_label.call_count == 0


async def test_pr_v2_set_head_of_queue_label_not_configured() -> None:
    """
    Without merge.head_of_queue_label nothing happens.
    """
    client = create_client()
    event = create_event()
    assert isinstance(event.config, V1)
    assert event.config.merge.head_of_queue_label is None
    pr_v2 = create_prv2(event=event, client=client)
    await pr_v2.set_head_of_queue_label(present=True)
    await pr_v2.set_head_of_queue_label(present=False)
    assert client.add_label.call_count == 0
    assert client.delete_label.call_count == 0


async def test_pr_v2_set_head_of_queue_label_api_error_is_not_fatal() -> None:
    """
    A failed label call must not interrupt merging, so no exception escapes.
    """
    client = create_client()
    client.add_label.response = create_response(
        content=b"<html>Service Unavailable</html>", status_code=503
    )
    event = create_event()
    assert isinstance(event.config, V1)
    event.config.merge.head_of_queue_label = "merge-queue-head"
    event.pull_request.labels = []
    pr_v2 = create_prv2(event=event, client=client)
    await pr_v2.set_head_of_queue_label(present=True)
    assert client.add_label.call_count == 1


async def test_pr_v2_set_head_of_queue_label_not_added_to_merged_pr() -> None:
    """
    The merge queue processes a pull request once more after merging it;
    that pass must not label a merged (or closed) pull request. Removal
    still works, so a label left from the merge comes off.
    """
    client = create_client()
    client.add_label.response = create_response(content=b"", status_code=204)
    client.delete_label.response = create_response(content=b"", status_code=204)
    event = create_event()
    assert isinstance(event.config, V1)
    event.config.merge.head_of_queue_label = "merge-queue-head"
    event.pull_request.state = PullRequestState.MERGED
    event.pull_request.labels = []
    pr_v2 = create_prv2(event=event, client=client)
    await pr_v2.set_head_of_queue_label(present=True)
    assert client.add_label.call_count == 0

    event.pull_request.labels = ["merge-queue-head"]
    await pr_v2.set_head_of_queue_label(present=False)
    assert client.delete_label.call_count == 1
