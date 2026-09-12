"""Provider-contract tests: no external Gemini request is made."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.agent import Intent, SupportAgent


class GeminiProviderTest(unittest.TestCase):
    def test_gemini_structured_classification_and_grounded_draft(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "SUPPORT_AGENT_OFFLINE": "0"}, clear=False):
            agent = SupportAgent(api_key="test-key", model="gemini-test")
            calls: list[tuple[str, bool]] = []

            def fake_gemini(prompt: str, schema=None, temperature: float = 0) -> str:
                calls.append((prompt, schema is not None))
                if schema is not None:
                    return '{"intent":"Delivery Delay","confidence":0.91,"rationale":"Late delivery."}'
                return "Sorry your package is delayed. Please check tracking for the latest update."

            agent._gemini = fake_gemini  # type: ignore[method-assign]
            result = agent.respond("My parcel is late and tracking has not changed.")

        self.assertEqual(agent.provider, "gemini")
        self.assertEqual(result.mode, "gemini")
        self.assertEqual(result.intent.intent, Intent.DELIVERY_DELAY)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(calls[0][1])
        self.assertIn("Historical resolutions", calls[1][0])


if __name__ == "__main__":
    unittest.main()
