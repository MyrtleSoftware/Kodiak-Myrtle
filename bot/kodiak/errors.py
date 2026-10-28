class RetryForSkippableChecks(Exception):
    pass


class PollForever(Exception):
    pass


class ApiCallException(Exception):
    def __init__(self, method: str, http_status_code: int, response: bytes) -> None:
        self.method = method
        self.status_code = http_status_code
        self.response = response


class GitHubApiInternalServerError(Exception):
    pass


class MergeFailed(Exception):
    """
    GitHub ran the merge in the background and reported that it failed, so
    nothing was merged.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
