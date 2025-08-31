from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

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

    @patch("verbatim.server.transcribe_file", return_value="hello world")
    async def test_transcriptions_endpoint(self, mock_transcribe):
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        resp = await self.client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        payload = await resp.json()
        self.assertEqual(payload["text"], "hello world")
        mock_transcribe.assert_called()

    @patch("verbatim.server.iterate_transcription", return_value=iter(["hi ", "there"]))
    async def test_transcriptions_stream_endpoint(self, mock_iter):
        data = aiohttp.FormData()
        data.add_field("file", b"abc", filename="a.wav", content_type="audio/wav")
        data.add_field("stream", "true")
        resp = await self.client.post("/audio/transcriptions", data=data)
        self.assertEqual(resp.status, 200)
        body = await resp.text()
        self.assertIn("transcript.text.delta", body)
        self.assertIn("hi ", body)
        self.assertIn("there", body)
        self.assertIn("transcript.text.done", body)
        mock_iter.assert_called()
