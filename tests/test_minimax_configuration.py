import ast
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from utils.chat_settings import MINIMAX_MODELS


ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("https://api.minimax.io/v1", "https://api.minimaxi.com/v1")


class MiniMaxConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.next_message = {"role": "assistant", "content": "answer"}
        self.environment = patch.dict(os.environ, {"API_KEY": "test-placeholder"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.dotenv = patch("dotenv.load_dotenv", return_value=False)
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo"):
            guard = patch(target, side_effect=AssertionError("Live network is forbidden in this test"))
            guard.start()
            self.addCleanup(guard.stop)
        self.http = patch("openai.OpenAI", side_effect=self.create_client)
        self.http.start()
        self.addCleanup(self.http.stop)

    def create_client(self, **kwargs):
        client = httpx.Client(transport=httpx.MockTransport(self.respond), trust_env=False)
        self.addCleanup(client.close)
        return OpenAI(**(kwargs | {"http_client": client}))

    def respond(self, request):
        self.requests.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, request=request, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": self.requests[-1][1]["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": self.next_message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    def load_text(self):
        return importlib.reload(importlib.import_module("utils.text2text_chat"))

    def load_langchain(self):
        return importlib.reload(importlib.import_module("utils.langchain_chat"))

    def load_robot_model(self):
        # Execute the real model configuration without launching the Panel UI
        # or importing the optional weather/search tools.
        path = ROOT / "utils/chat_robot.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"API_KEY", "BASE_URL", "CHAT_MODEL_NAME", "chat_model"}
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
                nodes.append(node)
            elif isinstance(node, ast.ImportFrom) and node.module in (
                "utils.chat_settings", "utils.minimax_chat"
            ):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                assigned = {child.id for target in node.targets for child in ast.walk(target)
                            if isinstance(child, ast.Name)}
                if assigned & names:
                    nodes.append(node)
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return namespace["chat_model"]

    def assert_minimax_defaults(self, body):
        self.assertEqual(body["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", body)
        for field in ("enable_thinking", "thinking", "reasoning_effort", "reasoning_split", "service_tier"):
            self.assertNotIn(field, body)

    def test_direct_default_requests_in_both_regions(self):
        for base_url in REGIONS:
            with self.subTest(base_url=base_url):
                os.environ["BASE_URL"] = base_url
                self.assertEqual(self.load_text().get_completions("hello"), "answer")
                url, body = self.requests[-1]
                self.assertEqual(url, base_url + "/chat/completions")
                self.assertEqual(body["model"], MINIMAX_MODELS[0])
                self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
                self.assert_minimax_defaults(body)

    def test_saved_models_and_per_call_override_are_preserved(self):
        os.environ["BASE_URL"] = REGIONS[0]
        for model in ("MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"):
            with self.subTest(model=model):
                os.environ["MODEL_NAME"] = model
                text = self.load_text()
                text.get_completions("hello")
                self.assertEqual(self.requests[-1][1]["model"], model)
                text.get_completions("hello", model_endpoint="per-call-model", max_tokens=16384)
                self.assertEqual(self.requests[-1][1]["model"], "per-call-model")
                self.assertEqual(self.requests[-1][1]["max_tokens"], 16384)

    def test_existing_default_and_gateway_configuration(self):
        self.load_text().get_completions("hello")
        url, body = self.requests[-1]
        self.assertEqual(url, "https://api.siliconflow.cn/v1/chat/completions")
        self.assertEqual(body["model"], "Qwen/Qwen3-8B")
        self.assertFalse(body["enable_thinking"])
        for base_url in ("https://gateway.example/v1", "https://api.minimax.io.example/v1"):
            os.environ["BASE_URL"] = base_url
            self.load_text().get_completions("hello")
            self.assertEqual(self.requests[-1][1]["model"], "Qwen/Qwen3-8B")
        os.environ["MODEL_NAME"] = "custom-route"
        self.load_text().get_completions("hello")
        self.assertEqual(self.requests[-1][1]["model"], "custom-route")

    def test_multimodal_parts_are_not_rewritten(self):
        os.environ["BASE_URL"] = REGIONS[0]
        content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            {"type": "video_url", "video_url": {"url": "mm_file://example"}},
        ]
        self.load_text().get_completions(content)
        self.assertEqual(self.requests[-1][1]["messages"][0]["content"], content)

    def test_both_langchain_entry_points_and_model_precedence(self):
        for base_url in REGIONS:
            os.environ["BASE_URL"] = base_url
            for load in (lambda: self.load_langchain().chat_model, self.load_robot_model):
                for settings, expected in (
                    ({}, MINIMAX_MODELS[0]),
                    ({"MODEL_NAME": "MiniMax-M3"}, "MiniMax-M3"),
                    ({"MODEL_NAME": "MiniMax-M2.7"}, "MiniMax-M2.7"),
                    ({"MODEL_NAME": "MiniMax-M3", "CHAT_MODEL_NAME": "saved-chat-model"}, "saved-chat-model"),
                ):
                    with self.subTest(base_url=base_url, settings=settings, load=load):
                        with patch.dict(os.environ, settings):
                            model = load()
                            self.assertEqual(model.invoke("hello").content, "answer")
                            url, body = self.requests[-1]
                            self.assertEqual(url, base_url + "/chat/completions")
                            self.assertEqual(body["model"], expected)
                            self.assert_minimax_defaults(body)

    def test_embeddings_and_other_provider_defaults_are_unchanged(self):
        module = self.load_langchain()
        module.chat_model.invoke("hello")
        self.assertEqual(self.requests[-1][1]["model"], "deepseek-ai/DeepSeek-V3")
        self.assertFalse(self.requests[-1][1]["enable_thinking"])
        self.load_robot_model().invoke("hello")
        self.assertEqual(self.requests[-1][1]["model"], "Qwen/Qwen3-8B")
        os.environ["BASE_URL"] = REGIONS[0]
        os.environ["EMBEDDING_API_KEY"] = "embedding-test-placeholder"
        module = self.load_langchain()
        self.assertEqual(module.embedding_model.openai_api_key.get_secret_value(), "embedding-test-placeholder")
        self.assertEqual(module.embedding_model.openai_api_base, "https://api.siliconflow.cn/v1")
        self.assertEqual(module.embedding_model.model, "BAAI/bge-m3")

    def test_tool_round_trip_preserves_reasoning_and_full_history(self):
        os.environ["BASE_URL"] = REGIONS[0]
        model = self.load_robot_model()
        tools = [{"type": "function", "function": {
            "name": "lookup", "description": "Look up a value",
            "parameters": {"type": "object", "properties": {}},
        }}]
        tool_calls = [{"id": "call-test", "type": "function", "function": {
            "name": "lookup", "arguments": "{}",
        }}]
        self.next_message = {
            "role": "assistant", "content": "<think>Keep this too</think>",
            "reasoning_content": "Keep the complete reasoning", "tool_calls": tool_calls,
        }
        first = model.bind(tools=tools).invoke("hello")
        self.assertEqual(first.additional_kwargs["reasoning_content"], self.next_message["reasoning_content"])
        self.assertEqual(self.requests[-1][1]["tools"], tools)
        self.next_message = {"role": "assistant", "content": "done"}
        history = [HumanMessage(content="hello"), first, ToolMessage(content="result", tool_call_id="call-test")]
        # Runnable.stream is used by agents; this adapter deliberately requests
        # full responses so the pinned SDK cannot drop streamed reasoning.
        self.assertEqual(list(model.bind(tools=tools).stream(history))[0].content, "done")
        body = self.requests[-1][1]
        self.assertFalse(body.get("stream", False))
        self.assertEqual(body["messages"][1]["reasoning_content"], "Keep the complete reasoning")
        self.assertEqual(body["messages"][1]["content"], "<think>Keep this too</think>")
        self.assertEqual(body["messages"][1]["tool_calls"], tool_calls)
        self.assertEqual(body["messages"][2], {"role": "tool", "content": "result", "tool_call_id": "call-test"})

    def test_current_catalog_and_documented_defaults(self):
        guide = (ROOT / "Env-Setup.md").read_text(encoding="utf-8")
        self.assertEqual(MINIMAX_MODELS, ("MiniMax-M3.1-Flash", "MiniMax-M3"))
        self.assertIn("MODEL_NAME=MiniMax-M3.1-Flash", guide)
        self.assertNotIn("| `MiniMax-M2", guide)
        for endpoint in REGIONS:
            self.assertIn(endpoint, guide)
        self.assertIn("$0.15 | $0.60 | $0.015", guide)
        self.assertIn("at most 512k input tokens | $0.30 | $1.20 | $0.06", guide)
        self.assertIn("above 512k input tokens | $0.60 | $2.40 | $0.12", guide)


if __name__ == "__main__":
    unittest.main()
