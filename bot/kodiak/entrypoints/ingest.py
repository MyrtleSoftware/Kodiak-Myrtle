"""
Accept webhooks from GitHub and add them to the Redis queues.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from typing import Any

import structlog
import uvicorn
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from starlette import status
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from kodiak import app_config as conf
from kodiak.entrypoints.worker import PubsubIngestQueueSchema
from kodiak.logging import configure_logging
from kodiak.queue import INGEST_QUEUE_NAMES, QUEUE_PUBSUB_INGEST, get_ingest_queue
from kodiak.redis_client import redis_bot
from kodiak.schemas import RawWebhookEvent
from kodiak.startup_refresh import refresh_pull_requests_with_retry

configure_logging()

logger = structlog.get_logger()

app = Starlette()
app.add_middleware(SentryAsgiMiddleware)


@app.route("/", methods=["GET"])
async def root(_: Request) -> Response:
    return PlainTextResponse("OK")


@app.route("/api/github/hook", methods=["POST"])
async def github_webhook_event(request: Request) -> Response:
    expected_sha = hmac.new(
        key=conf.SECRET_KEY.encode(), msg=(await request.body()), digestmod=hashlib.sha1
    ).hexdigest()

    try:
        github_event = request.headers["X-Github-Event"]
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missing required X-Github-Event header",
        ) from e
    try:
        hub_signature = request.headers["X-Hub-Signature"]
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missing required X-Hub-Signature header",
        ) from e

    sha = hub_signature.replace("sha1=", "")
    if not hmac.compare_digest(sha, expected_sha):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature: X-Hub-Signature",
        )
    event: dict[str, Any] = await request.json()

    log = logger.bind(event_name=github_event, event=event)
    installation_id: int | None = event.get("installation", {}).get("id")

    if github_event in {
        "github_app_authorization",
        "installation",
        "installation_repositories",
    }:
        log.info("administrative_event_received")
        return JSONResponse({"ok": True})

    if installation_id is None:
        log.warning("unexpected_event_skipped")
        return JSONResponse({"ok": True})

    await enqueue_event(installation_id, github_event, event)
    return JSONResponse({"ok": True})


async def enqueue_event(
    installation_id: int, event_name: str, payload: dict[str, Any]
) -> None:
    ingest_queue = get_ingest_queue(installation_id)

    await redis_bot.rpush(
        ingest_queue,
        RawWebhookEvent(event_name=event_name, payload=payload).json(),
    )

    # workers pop from the head, so drop the oldest events when the queue is full.
    await redis_bot.ltrim(ingest_queue, -conf.INGEST_QUEUE_LENGTH, -1)
    await redis_bot.sadd(INGEST_QUEUE_NAMES, ingest_queue)
    await redis_bot.publish(
        QUEUE_PUBSUB_INGEST,
        PubsubIngestQueueSchema(installation_id=installation_id).json(),
    )


async def refresh_when_serving(server: uvicorn.Server) -> None:
    """
    Look for pull requests whose webhooks we missed while we were down. Wait
    until uvicorn has bound its socket: from then on, GitHub's webhooks reach
    us, so the scan misses nothing.
    """
    while not server.started:
        await asyncio.sleep(0.1)
    await refresh_pull_requests_with_retry(enqueue_event)


async def serve(server: uvicorn.Server) -> None:
    refresh = asyncio.create_task(refresh_when_serving(server))
    try:
        await server.serve()
    finally:
        refresh.cancel()


if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=conf.PORT))
    server.config.setup_event_loop()
    asyncio.run(serve(server))
