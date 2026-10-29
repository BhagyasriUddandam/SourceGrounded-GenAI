import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from sourcegrounded.llm import ChatClient, ProviderError


class ProviderTests(unittest.TestCase):
    def test_missing_key_fails_before_network(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            client = ChatClient()
            with self.assertRaisesRegex(ProviderError, "Missing GROQ_API_KEY"):
                client.chat([{"role": "user", "content": "test"}])

    def test_non_https_remote_endpoint_blocked(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            ChatClient(base_url="http://remote.example.com/v1")

    def test_official_groq_sdk_chat_integration(self):
        """Exercises real SDK calling conventions using a controlled fake module."""
        calls = {}

        class FakeCompletions:
            def create(self, **kwargs):
                calls["params"] = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="Verified mock response"))],
                    usage=SimpleNamespace(model_dump=lambda: {"total_tokens":25}),
                    model="openai/gpt-oss-120b",
                )

        class FakeGroq:
            def __init__(self, **kwargs):
                calls["client"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        fake_sdk = SimpleNamespace(Groq=FakeGroq, APIError=RuntimeError)
        with patch.dict(sys.modules, {"groq": fake_sdk}):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-not-a-real-key", "GENAI_MODEL": "openai/gpt-oss-120b"}):
                result = ChatClient().chat([{"role": "user", "content": "Hello"}])
        self.assertEqual(result.text, "Verified mock response")
        self.assertEqual(result.usage["total_tokens"], 25)
        self.assertEqual(calls["params"]["model"], "openai/gpt-oss-120b")
        self.assertEqual(calls["params"]["reasoning_effort"], "low")
        self.assertEqual(calls["client"]["base_url"], "https://api.groq.com")


if __name__ == "__main__":
    unittest.main()
