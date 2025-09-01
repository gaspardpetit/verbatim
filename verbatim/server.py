import asyncio
import json
import os
import tempfile
import threading
from dataclasses import replace
from typing import Iterable, Optional, Union, cast

from aiohttp import web
from aiohttp.multipart import BodyPartReader

from .audio.sources.factory import create_audio_source
from .audio.sources.sourceconfig import SourceConfig
from .config import Config
from .verbatim import Verbatim


async def _handle_transcriptions(request: web.Request) -> web.StreamResponse:
    config: Config = request.app["config"]
    reader = await request.multipart()
    file_path: Optional[str] = None
    language: Optional[str] = None
    stream = False

    async for raw_part in reader:
        part = cast(BodyPartReader, raw_part)
        if part.name == "file":
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                while chunk := await part.read_chunk():
                    tmp.write(chunk)
                file_path = tmp.name
        elif part.name == "language":
            language = await part.text()
        elif part.name == "stream":
            stream = (await part.text()).lower() == "true"

    if file_path is None:
        return web.json_response({"error": "file is required"}, status=400)

    if not stream:
        transcribe_func = transcribe_file
        try:
            text = await asyncio.to_thread(transcribe_func, file_path, config, language)
        finally:
            os.unlink(file_path)
        return web.json_response({"text": text})

    # streaming response
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[Optional[Union[str, Exception]]] = asyncio.Queue()

    transcribe_iter = iterate_transcription

    def worker() -> None:
        try:
            for piece in transcribe_iter(file_path, config, language):
                asyncio.run_coroutine_threadsafe(queue.put(piece), loop)
        except Exception as exc:  # pragma: no cover - best effort  # pylint: disable=broad-exception-caught
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            os.unlink(file_path)

    threading.Thread(target=worker, daemon=True).start()

    pieces: list[str] = []
    error: Optional[Exception] = None
    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            error = item
            await resp.write(
                f"data: {json.dumps({'type': 'error', 'error': str(item)})}\n\n".encode()
            )
            break
        piece = cast(str, item)
        pieces.append(piece)
        await resp.write(
            f"data: {json.dumps({'type': 'transcript.text.delta', 'delta': piece})}\n\n".encode()
        )

    if error is None:
        final_text = "".join(pieces).strip()
        await resp.write(
            f"data: {json.dumps({'type': 'transcript.text.done', 'text': final_text})}\n\n".encode()
        )
    await resp.write_eof()
    return resp


def iterate_transcription(
    path: str, base_config: Config, language: Optional[str] = None
) -> Iterable[str]:
    cfg = replace(base_config)
    if language:
        cfg.lang = [language]
    basename = os.path.splitext(os.path.basename(path))[0]
    working_prefix = os.path.join(cfg.working_dir, basename)
    output_prefix = os.path.join(cfg.output_dir, basename)
    source_config = SourceConfig()
    audio_source = create_audio_source(
        source_config=source_config,
        device=cfg.device,
        input_source=path,
        start_time="00:00.000",
        stop_time="",
        working_prefix_no_ext=working_prefix,
        output_prefix_no_ext=output_prefix,
        stream=cfg.stream,
    )
    transcriber = Verbatim(cfg)
    with audio_source.open() as audio_stream:
        for utterance, _, _ in transcriber.transcribe(
            audio_stream=audio_stream, working_prefix_no_ext=working_prefix
        ):
            yield utterance.text


def transcribe_file(path: str, base_config: Config, language: Optional[str] = None) -> str:
    return "".join(iterate_transcription(path, base_config, language)).strip()


AVAILABLE_MODELS = [
    {
        "id": "whisper-large-v3-verbatim",
        "object": "model",
        "created": 0,
        "owned_by": "verbatim",
    }
]


async def _handle_models(_request: web.Request) -> web.Response:
    return web.json_response({"data": AVAILABLE_MODELS, "object": "list"})


async def _handle_model(request: web.Request) -> web.Response:
    model_id = request.match_info["model_id"]
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return web.json_response(model)
    return web.json_response({"error": "model not found"}, status=404)


def create_app(config: Config) -> web.Application:
    app = web.Application()
    app["config"] = config
    app.router.add_post("/audio/transcriptions", _handle_transcriptions)
    app.router.add_get("/models", _handle_models)
    app.router.add_get("/models/{model_id}", _handle_model)
    return app


def serve(config: Config, host: str = "0.0.0.0", port: int = 8000) -> None:  # nosec B104
    app = create_app(config)
    web.run_app(app, host=host, port=port)
