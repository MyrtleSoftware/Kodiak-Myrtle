from typing import Any, Dict, List, Optional, Tuple

import pydantic
import structlog

logger = structlog.get_logger()


class CommitConnection(pydantic.BaseModel):
    totalCount: int


class User(pydantic.BaseModel):
    databaseId: Optional[int]
    login: str
    name: Optional[str]
    type: str

    def __hash__(self) -> int:
        # defining a hash method allows us to deduplicate CommitAuthors easily.
        return hash(self.databaseId) + hash(self.login) + hash(self.name)


class GitActor(pydantic.BaseModel):
    user: Optional[User]


class Commit(pydantic.BaseModel):
    parents: CommitConnection
    author: Optional[GitActor]


class PullRequestCommit(pydantic.BaseModel):
    commit: Commit


class PullRequestCommitConnection(pydantic.BaseModel):
    nodes: Optional[List[PullRequestCommit]]
    totalCount: Optional[int] = None


class PullRequest(pydantic.BaseModel):
    commitHistory: PullRequestCommitConnection


class PageInfo(pydantic.BaseModel):
    hasNextPage: bool
    endCursor: Optional[str]


class PullRequestCommitPage(pydantic.BaseModel):
    nodes: Optional[List[PullRequestCommit]]
    pageInfo: PageInfo


def get_commit_history(*, pr: Dict[str, Any]) -> Tuple[List[Commit], Optional[int]]:
    """
    The pull request's commits (the last 100, see the commitHistory query) and
    its total commit count, which tells whether that list is complete.
    """
    try:
        pull_request = PullRequest.parse_obj(pr)
    except pydantic.ValidationError:
        logger.exception("problem parsing commit authors")
        return [], None
    nodes = pull_request.commitHistory.nodes
    commits = [node.commit for node in nodes] if nodes else []
    return commits, pull_request.commitHistory.totalCount


def get_commits(*, pr: Dict[str, Any]) -> List[Commit]:
    """
    Extract the commit authors from the pull request commits.
    """
    return get_commit_history(pr=pr)[0]


def parse_commit_page(
    *, connection: Dict[str, Any]
) -> Optional[Tuple[List[Commit], PageInfo]]:
    """
    One page of a pull request's commits connection, as returned by
    GET_PULL_REQUEST_COMMITS_QUERY.
    """
    try:
        page = PullRequestCommitPage.parse_obj(connection)
    except pydantic.ValidationError:
        logger.exception("problem parsing commit page")
        return None
    commits = [node.commit for node in page.nodes] if page.nodes else []
    return commits, page.pageInfo
