"""Transparent comparison baselines used by the evaluation runner."""
from __future__ import annotations

from src.agent import AgentResponse, EscalationResult, Intent, IntentResult, SupportAgent


class MajorityBaseline:
    """Always predicts the training-set majority intent and no escalation."""
    def __init__(self, majority: Intent = Intent.GENERAL_INQUIRY) -> None:
        self.majority = majority

    def respond(self, _: str) -> AgentResponse:
        return AgentResponse(intent=IntentResult(intent=self.majority, confidence=1.0, rationale="Majority-class baseline."),
                             reply="Thanks for contacting support. Please visit our Help page for assistance.",
                             escalation=EscalationResult(escalate=False, escalation_reason="Baseline does not assess risk."), mode="mock")


class SimpleBaseline(SupportAgent):
    """Keyword-based zero-shot proxy; intentionally omits safeguards for comparison."""
    def respond(self, text: str) -> AgentResponse:
        result = self.classify(text)
        return AgentResponse(intent=result, reply="Thanks for reaching out. Please check the Help page for next steps.",
                             escalation=EscalationResult(escalate=False, escalation_reason="Simple baseline does not escalate."), mode="mock")
