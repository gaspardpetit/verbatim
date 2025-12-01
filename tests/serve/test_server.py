from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

from verbatim.config import Config
from verbatim_serve.server import create_app


class ServerEndpointTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = Config(device="cpu", output_dir=".", working_dir=".", stream=False, offline=True)
        self.client: TestClient | None = None
        self.server: TestServer | None = None

    async def asyncTearDown(self):
        if self.client:
            await self.client.close()
        self.client = None
        self.server = None

    async def _start_client(self, *, transcribe_func=None, transcribe_iter=None):
        app = create_app(self.config, transcribe_func=transcribe_func, transcribe_iter=transcribe_iter)
        self.server = TestServer(app)
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def test_transcriptions_endpoint(self):
        mock_transcribe = MagicMock(return_value="hello world")
        await self._start_client(transcribe_func=mock_transcribe)
        client = self.client
        self.assertIsNotNone(client)
        client = cast(TestClient, client)
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        resp = await client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertEqual(payload["text"], "hello world")
        mock_transcribe.assert_called()

    async def test_transcriptions_stream_endpoint(self):
        mock_iter = MagicMock(return_value=iter(["hi ", "there"]))
        await self._start_client(transcribe_iter=mock_iter)
        client = self.client
        self.assertIsNotNone(client)
        client = cast(TestClient, client)
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        data.add_field("stream", "true")
        resp = await client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn("transcript.text.delta", body)
        self.assertIn("hi ", body)
        self.assertIn("there", body)
        self.assertIn("transcript.text.done", body)
        mock_iter.assert_called()

    async def test_transcriptions_stream_endpoint_error(self):
        mock_iter = MagicMock(side_effect=RuntimeError("boom"))
        await self._start_client(transcribe_iter=mock_iter)
        client = self.client
        self.assertIsNotNone(client)
        client = cast(TestClient, client)
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        data.add_field("stream", "true")
        resp = await client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn('"type": "error"', body)
        self.assertNotIn("transcript.text.done", body)
        mock_iter.assert_called()

    async def test_models_endpoints(self):
        await self._start_client()
        client = self.client
        self.assertIsNotNone(client)
        client = cast(TestClient, client)
        resp = await client.get("/models")
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertTrue(any(m["id"] == "whisper-large-v3-verbatim" for m in payload["data"]))

        resp = await client.get("/models/whisper-large-v3-verbatim")
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertEqual(payload["id"], "whisper-large-v3-verbatim")

        resp = await client.get("/models/missing")
        self.assertEqual(resp.status, 404)
