from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

from verbatim.config import Config
from verbatim.server import create_app


class ServerEndpointTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = Config(device="cpu", output_dir=".", working_dir=".", stream=False, offline=True)
        app = create_app(config)
        self.server = TestServer(app)
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_transcriptions_endpoint(self):
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        mock_transcribe = MagicMock(return_value="hello world")
        self.server.app["transcribe_func"] = mock_transcribe
        resp = await self.client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertEqual(payload["text"], "hello world")
        mock_transcribe.assert_called()

    async def test_transcriptions_stream_endpoint(self):
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        data.add_field("stream", "true")
        mock_iter = MagicMock(return_value=iter(["hi ", "there"]))
        self.server.app["transcribe_iter"] = mock_iter
        resp = await self.client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn("transcript.text.delta", body)
        self.assertIn("hi ", body)
        self.assertIn("there", body)
        self.assertIn("transcript.text.done", body)
        mock_iter.assert_called()

    async def test_transcriptions_stream_endpoint_error(self):
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        data.add_field("stream", "true")
        mock_iter = MagicMock(side_effect=RuntimeError("boom"))
        self.server.app["transcribe_iter"] = mock_iter
        resp = await self.client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn('"type": "error"', body)
        self.assertNotIn("transcript.text.done", body)
        mock_iter.assert_called()

    async def test_models_endpoints(self):
        resp = await self.client.get("/models")
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertTrue(any(m["id"] == "whisper-large-v3-verbatim" for m in payload["data"]))

        resp = await self.client.get("/models/whisper-large-v3-verbatim")
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertEqual(payload["id"], "whisper-large-v3-verbatim")

        resp = await self.client.get("/models/missing")
        self.assertEqual(resp.status, 404)
