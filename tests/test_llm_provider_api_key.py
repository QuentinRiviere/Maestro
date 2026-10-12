"""Contract tests for LLM-provider credential resolution."""

from __future__ import annotations

import os
import sys
import unittest


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

from services import llm_service


class TestProviderApiKey(unittest.TestCase):
    def setUp(self):
        self.services = {
            "llm_remote_api_key": "sk-remote",
            "openai_api_key": "sk-openai",
            "anthropic_api_key": "sk-anthropic",
        }

    def test_each_provider_gets_only_its_own_credential(self):
        for provider, expected in (
            ("remote", "sk-remote"),
            ("openai", "sk-openai"),
            ("anthropic", "sk-anthropic"),
        ):
            with self.subTest(provider=provider):
                self.assertEqual(
                    llm_service.provider_api_key(provider, self.services),
                    expected,
                )

    def test_local_unknown_and_unconfigured_providers_send_no_key(self):
        for provider in ("local", "", "bogus", None):
            with self.subTest(provider=provider):
                self.assertEqual(
                    llm_service.provider_api_key(provider, self.services),
                    "",
                )
        self.assertEqual(llm_service.provider_api_key("remote", {}), "")

    def test_openrouter_key_falls_back_to_environment(self):
        from unittest import mock

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env"}, clear=False):
            self.assertEqual(llm_service.provider_api_key("openrouter", {}), "sk-or-env")
            self.assertEqual(
                llm_service.provider_api_key("openrouter", {"openrouter_api_key": "sk-or-saved"}),
                "sk-or-saved",
            )
            # The environment key never leaks to another provider.
            self.assertEqual(llm_service.provider_api_key("openai", {}), "")

    def test_hosted_providers_have_default_base_urls(self):
        self.assertEqual(llm_service.provider_base_url("openrouter"), "https://openrouter.ai/api")
        self.assertEqual(llm_service.provider_base_url("openai"), "https://api.openai.com")
        self.assertEqual(llm_service.provider_base_url("openrouter", "http://proxy:8080"), "http://proxy:8080")
        self.assertEqual(llm_service.provider_base_url("remote"), "")

    def test_every_credential_setting_is_persistable(self):
        launch_path = os.path.join(_APP, "launch.py")
        with open(launch_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        for setting in llm_service.PROVIDER_API_KEY_SETTING.values():
            with self.subTest(setting=setting):
                self.assertIn(f'"{setting}"', source)

    def test_director_uses_the_shared_mapping(self):
        path = os.path.join(_APP, "services", "director_pipeline.py")
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("llm_service.provider_api_key(", source)


class TestApiHeaders(unittest.TestCase):
    def setUp(self):
        self.saved = (llm_service._provider, llm_service._api_key)

    def tearDown(self):
        llm_service._provider, llm_service._api_key = self.saved

    def _headers_for(self, provider, key):
        llm_service._provider = provider
        llm_service._api_key = key
        return llm_service._api_headers()

    def test_remote_uses_bearer_token(self):
        self.assertEqual(
            self._headers_for("remote", "sk-remote").get("Authorization"),
            "Bearer sk-remote",
        )

    def test_anthropic_uses_its_native_headers(self):
        headers = self._headers_for("anthropic", "sk-anthropic")
        self.assertEqual(headers.get("x-api-key"), "sk-anthropic")
        self.assertEqual(headers.get("anthropic-version"), "2023-06-01")
        self.assertNotIn("Authorization", headers)

    def test_local_and_keyless_remote_send_no_auth(self):
        self.assertNotIn("Authorization", self._headers_for("local", ""))
        self.assertNotIn("Authorization", self._headers_for("remote", ""))


class TestOpenRouterReasoning(unittest.TestCase):
    def finalize(self, provider, payload):
        saved = (llm_service._provider, llm_service._model_id)
        llm_service._provider, llm_service._model_id = provider, "openrouter/free"
        try:
            return llm_service._finalize_payload(dict(payload))
        finally:
            llm_service._provider, llm_service._model_id = saved

    def test_thinking_off_bounds_reasoning_and_keeps_the_answer_budget(self):
        prepared = self.finalize("openrouter", {"max_tokens": 768, "enable_thinking": False, "messages": []})
        self.assertEqual(prepared["reasoning"], {"effort": "low", "exclude": True})
        self.assertEqual(prepared["max_tokens"], 768 + llm_service.OPENROUTER_REASONING_HEADROOM)

    def test_thinking_on_and_other_providers_are_untouched(self):
        on = self.finalize("openrouter", {"max_tokens": 768, "enable_thinking": True, "messages": []})
        self.assertNotIn("reasoning", on)
        openai = self.finalize("openai", {"max_tokens": 768, "enable_thinking": False, "messages": []})
        self.assertNotIn("reasoning", openai)
        self.assertEqual(openai["max_tokens"], 768)


if __name__ == "__main__":
    unittest.main(verbosity=2)
