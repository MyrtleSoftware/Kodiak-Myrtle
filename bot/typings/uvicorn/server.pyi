from uvicorn.config import Config

class Server:
    config: Config
    started: bool
    def __init__(self, config: Config) -> None: ...
    async def serve(self) -> None: ...
