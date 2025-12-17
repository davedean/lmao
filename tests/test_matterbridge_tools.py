import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import urllib.error

from lmao.matterbridge import MatterbridgeConfig, configure_matterbridge
from lmao.plugins import discover_plugins
from lmao.tools import ToolCall, run_tool
from lmao.matterbridge_common import _resolve_matterbridge_config, ToolError


class MatterbridgeCommonTests(TestCase):
    def test_resolves_cli_config_env_precedence(self) -> None:
        context = MatterbridgeConfig(uri="http://cfg.uri", gateway="cfg-gateway")
        with patch.dict(
            os.environ,
            {"MATTERBRIDGE_URI": "http://env.uri", "MATTERBRIDGE_GATEWAY": "env-gateway"},
            clear=True,
        ):
            uri, gateway = _resolve_matterbridge_config(context, {"gateway": "override"}, require_gateway=True)
            self.assertEqual("http://cfg.uri", uri)
            self.assertEqual("override", gateway)

            uri, gateway = _resolve_matterbridge_config(MatterbridgeConfig(uri=None, gateway=None), {}, require_gateway=False)
            self.assertEqual("http://env.uri", uri)
            self.assertEqual("env-gateway", gateway)

        with patch.dict(os.environ, {"MATTERBRIDGE_URI": "http://env.uri"}, clear=True):
            with self.assertRaises(ToolError):
                _resolve_matterbridge_config(MatterbridgeConfig(uri=None, gateway=None), {}, require_gateway=True)


class MatterbridgePluginTests(TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp_dir.name).resolve()
        tools_dir = Path(__file__).resolve().parent.parent / "lmao" / "tools"
        self.plugins = discover_plugins([tools_dir], self.base, allow_outside_base=True)
        configure_matterbridge("http://matterbridge.local", "default-gateway")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _run_tool(self, tool: str, args: Any) -> dict[str, Any]:
        call = ToolCall(tool=tool, target="", args=args)
        output = run_tool(
            call,
            base=self.base,
            extra_roots=[],
            skill_roots=[],
            yolo_enabled=False,
            plugin_tools=self.plugins,
        )
        return json.loads(output)

    def test_send_string_payload(self) -> None:
        class DummyResponse:
            def __init__(self, value: Any) -> None:
                self._data = json.dumps(value).encode("utf-8")

            def __enter__(self) -> "DummyResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return False

            def read(self) -> bytes:
                return self._data

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = DummyResponse({"status": "ok"})
            payload = self._run_tool("matterbridge_send", "Hello from agent")
        self.assertTrue(payload["success"])
        self.assertEqual("default-gateway", payload["data"]["gateway"])
        self.assertEqual("ok", payload["data"]["response"]["status"])

    def test_send_dict_payload_overrides_gateway(self) -> None:
        recorded: dict[str, Any] = {}

        class DummyResponse:
            def __init__(self) -> None:
                self._data = json.dumps({"status": "ok"}).encode("utf-8")

            def __enter__(self) -> "DummyResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return False

            def read(self) -> bytes:
                return self._data

        def fake_urlopen(request, timeout):
            recorded["timeout"] = timeout
            recorded["body"] = json.loads(request.data.decode("utf-8"))
            return DummyResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            payload = self._run_tool(
                "matterbridge_send",
                {"text": "Override message", "gateway": "gateway-override", "username": "status-bot", "timeout": 3},
            )
        self.assertTrue(payload["success"])
        self.assertEqual("gateway-override", payload["data"]["gateway"])
        self.assertEqual("status-bot", recorded["body"]["username"])
        self.assertEqual("gateway-override", recorded["body"]["gateway"])
        self.assertEqual(3, recorded["timeout"])

    def test_send_missing_text_returns_error(self) -> None:
        payload = self._run_tool("matterbridge_send", {"gateway": "gateway-override"})
        self.assertFalse(payload["success"])
        self.assertIn("text", payload["error"])

    def test_send_http_error(self) -> None:
        error = urllib.error.HTTPError(
            "http://matterbridge.local/api/message",
            500,
            "Server Error",
            hdrs=None,
            fp=io.BytesIO(b"broken"),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            payload = self._run_tool("matterbridge_send", "Try again")
        self.assertFalse(payload["success"])
        self.assertIn("500", payload["error"])

    def test_read_buffer_mode(self) -> None:
        messages = [{"text": "first"}, {"text": "second"}]

        class DummyResponse:
            def __init__(self) -> None:
                self._data = json.dumps(messages).encode("utf-8")

            def __enter__(self) -> "DummyResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return False

            def read(self) -> bytes:
                return self._data

        with patch("urllib.request.urlopen", return_value=DummyResponse()):
            payload = self._run_tool("matterbridge_read", {"limit": 1})
        self.assertTrue(payload["success"])
        self.assertEqual("buffer", payload["data"]["via"])
        self.assertEqual(1, len(payload["data"]["messages"]))
        self.assertEqual("default-gateway", payload["data"]["gateway"])

    def test_read_stream_mode_handles_events_and_limit(self) -> None:
        class DummyStreamResponse:
            def __init__(self) -> None:
                self._lines = [
                    b'{"event":"api_connected"}\n',
                    b'{"text":"hello"}\n',
                    b'{"text":"world"}\n',
                ]
                self._index = 0

            def __enter__(self) -> "DummyStreamResponse":
                return self

            def __exit__(self, *args: Any) -> None:
                return False

            def readline(self) -> bytes:
                if self._index >= len(self._lines):
                    return b""
                line = self._lines[self._index]
                self._index += 1
                return line

        with patch("urllib.request.urlopen", return_value=DummyStreamResponse()):
            payload = self._run_tool("matterbridge_read", {"stream": True, "limit": 1})
        self.assertTrue(payload["success"])
        self.assertEqual("stream", payload["data"]["via"])
        self.assertTrue(payload["data"]["truncated"])
        self.assertEqual(1, len(payload["data"]["messages"]))

    def test_read_buffer_timeout_surfaces_tool_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timed out")):
            payload = self._run_tool("matterbridge_read", {"limit": 1})
        self.assertFalse(payload["success"])
        self.assertIn("network error", payload["error"])
