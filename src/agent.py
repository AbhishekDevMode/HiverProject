"""Typed support-agent pipeline with deterministic and OpenAI-backed modes."""
from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Intent(str, Enum):
    ORDER_TRACKING = "Order Tracking"
    REFUND_BILLING = "Refund/Billing"
    ACCOUNT_AUTH = "Account Auth"
    TECHNICAL_ISSUE = "Technical Issue"
    GENERAL_INQUIRY = "General Inquiry"
    DELIVERY_DELAY = "Delivery Delay"
    MISSING_PACKAGE = "Missing Package"
    CANCELLATION = "Cancellation"

class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str


class EscalationResult(BaseModel):
    escalate: bool
    escalation_reason: str


class AgentResponse(BaseModel):
    intent: IntentResult
    reply: str
    escalation: EscalationResult
    mode: Literal["mock", "openai", "gemini"]


class HistoricalResolutionRetriever:
    """Small local retrieval index over linked, historical brand resolutions."""
    def __init__(self, path: Path = Path("data/subsample.csv")) -> None:
        self.examples: list[tuple[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if path.exists():
            data = pd.read_csv(path, usecols=lambda column: column in {"incoming_text", "historical_resolution"})
            data = data.dropna().loc[lambda frame: frame.get("historical_resolution", pd.Series(dtype=str)).astype(str).str.len() > 20]
            self.examples = list(zip(data["incoming_text"].astype(str), data["historical_resolution"].astype(str)))
        if self.examples:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5_000)
            self.matrix = self.vectorizer.fit_transform([item[0] for item in self.examples])

    def retrieve(self, text: str, k: int = 2) -> list[str]:
        if not self.vectorizer or self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([text]), self.matrix).ravel()
        return [self.examples[index][1] for index in scores.argsort()[-k:][::-1] if scores[index] > 0]


PII = re.compile(r"\b(?:\d[ -]*?){13,16}\b|\b(?:password|passcode|cvv|otp|one.time code)\b", re.I)
FRUSTRATION = re.compile(r"\b(?:ridiculous|useless|worst|hate|furious|again\?|unbelievable|sarcasm)\b", re.I)
MUTATION = re.compile(r"\b(?:refund|cancel|change address|replace|chargeback)\b", re.I)


class SupportAgent:
    """Gemini-first support agent with OpenAI and deterministic fallbacks."""
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        offline = os.getenv("SUPPORT_AGENT_OFFLINE", "").lower() in {"1", "true", "yes"}
        requested = os.getenv("LLM_PROVIDER", "").lower()
        if offline or requested == "mock":
            self.provider, self.api_key = "mock", None
        elif requested == "openai" or (not requested and not os.getenv("GEMINI_API_KEY") and os.getenv("OPENAI_API_KEY")):
            self.provider, self.api_key = "openai", api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.provider, self.api_key = "gemini", api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or (os.getenv("GEMINI_MODEL", "gemini-2.5-flash") if self.provider == "gemini" else os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        # Disable remote retries after a provider error: a 200-record evaluation
        # should finish safely even when a key has no remaining quota.
        self._provider_available = bool(self.api_key)
        self.provider_error: str | None = None
        self.retriever = HistoricalResolutionRetriever()

    def _gemini(self, prompt: str, schema: type[BaseModel] | None = None, temperature: float = 0) -> str:
        """Generate text/JSON through Google's supported Python SDK."""
        from google import genai
        config: dict[str, object] = {"temperature": temperature}
        if schema:
            config.update({"response_mime_type": "application/json", "response_json_schema": schema.model_json_schema()})
        response = genai.Client(api_key=self.api_key).models.generate_content(model=self.model, contents=prompt, config=config)
        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return response.text

    def classify(self, text: str) -> IntentResult:
        if self._provider_available and self.api_key:
            try:
                prompt = "Classify the customer-support message into exactly one allowed intent. Return the requested JSON only. Allowed intents: " + ", ".join(x.value for x in Intent) + "\nMessage: " + text
                if self.provider == "gemini":
                    return IntentResult.model_validate_json(self._gemini(prompt, IntentResult))
                from openai import OpenAI
                response = OpenAI(api_key=self.api_key).chat.completions.create(model=self.model, response_format={"type": "json_object"}, messages=[{"role": "system", "content": prompt}], temperature=0)
                return IntentResult.model_validate_json(response.choices[0].message.content or "{}")
            except Exception as error:
                self._provider_available = False
                self.provider_error = f"{type(error).__name__}: {error}"
        lower = text.lower()
        rules = [(Intent.MISSING_PACKAGE, ["says delivered", "not here", "missing package"]),
                 (Intent.DELIVERY_DELAY, ["delayed", "due yesterday", "late"]),
                 (Intent.REFUND_BILLING, ["charged", "refund", "billing", "payment"]),
                 (Intent.ACCOUNT_AUTH, ["sign in", "login", "password", "account"]),
                 (Intent.TECHNICAL_ISSUE, ["crash", "app", "error", "website"]),
                 (Intent.CANCELLATION, ["cancel"]),
                 (Intent.ORDER_TRACKING, ["where is", "track", "status", "order"])]
        for intent, words in rules:
            if any(word in lower for word in words):
                return IntentResult(intent=intent, confidence=0.87, rationale=f"Matched support pattern for {intent.value}.")
        return IntentResult(intent=Intent.GENERAL_INQUIRY, confidence=0.62, rationale="No specific pattern matched.")

    def generate_reply(self, text: str, result: IntentResult) -> str:
        examples = self.retriever.retrieve(text)
        if self._provider_available and self.api_key and examples:
            try:
                prompt = ("Draft one concise public @AmazonHelp reply. Use the historical examples only as style/resolution evidence; "
                          "never request personal data, promise a refund, or reveal account information. Escalate-sensitive requests must direct users to a secure channel. "
                          f"Intent: {result.intent.value}\nCustomer: {text}\nHistorical resolutions:\n" + "\n---\n".join(examples))
                if self.provider == "gemini":
                    draft = self._gemini(prompt, temperature=0.2)
                else:
                    from openai import OpenAI
                    response = OpenAI(api_key=self.api_key).chat.completions.create(model=self.model, messages=[{"role": "system", "content": "You write safe, concise customer-support replies."}, {"role": "user", "content": prompt}], temperature=0.2)
                    draft = response.choices[0].message.content
                if draft:
                    return draft.strip()
            except Exception as error:
                self._provider_available = False
                self.provider_error = f"{type(error).__name__}: {error}"
        templates = {
            Intent.ORDER_TRACKING: "I’m sorry you’re waiting. Please check the order’s tracking page for the latest carrier scan; if it has not updated for 48 hours, contact us with the order number by secure channel.",
            Intent.DELIVERY_DELAY: "I’m sorry the delivery is late. Tracking can update after the carrier’s next scan; please check it again within 24–48 hours. We can review the order securely if the delay continues.",
            Intent.REFUND_BILLING: "I’m sorry about the billing issue. For your security, please do not share payment details here. Contact support through the secure order page so a specialist can review the charge.",
            Intent.ACCOUNT_AUTH: "I’m sorry you’re having trouble signing in. Use the password-reset flow and never post a password or one-time code. If that does not work, contact account support through the secure help page.",
            Intent.TECHNICAL_ISSUE: "Sorry the experience is not working. Please update the app, restart it, and try again. If it persists, send the app version and a screenshot with personal details removed.",
            Intent.MISSING_PACKAGE: "I’m sorry the package is marked delivered but is not there. Please check safe locations and with neighbours, then report it through the order page so the delivery team can investigate.",
            Intent.CANCELLATION: "I can help point you in the right direction. Open the order page and select cancellation if the item has not shipped; a support specialist can confirm the available options.",
            Intent.GENERAL_INQUIRY: "Thanks for reaching out. Please share the order context through the secure help page (without posting personal details), and support can help with the next step.",
        }
        return templates[result.intent]

    def decide_escalation(self, text: str, intent: IntentResult) -> EscalationResult:
        reasons: list[str] = []
        if intent.confidence < .70: reasons.append("intent confidence is below 0.70")
        if PII.search(text): reasons.append("message may contain sensitive credentials or payment data")
        if FRUSTRATION.search(text): reasons.append("high frustration or sarcasm detected")
        if MUTATION.search(text): reasons.append("request requires a backend account action")
        return EscalationResult(escalate=bool(reasons), escalation_reason="; ".join(reasons) if reasons else "No mandatory escalation trigger detected.")

    def respond(self, text: str) -> AgentResponse:
        intent = self.classify(text)
        return AgentResponse(intent=intent, reply=self.generate_reply(text, intent), escalation=self.decide_escalation(text, intent), mode=self.provider if self._provider_available else "mock")
