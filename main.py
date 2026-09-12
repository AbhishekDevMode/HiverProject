from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

from src.agent import SupportAgent
from src.baselines import MajorityBaseline, SimpleBaseline
from src.eval_harness import evaluate, load_golden
from src.ingest import build_subsample


def line(name: str, report: dict) -> str:
    i, e = report["intent"], report["escalation"]
    return f"{name:<18} {i['accuracy']:>8.3f} {i['f1_macro']:>10.3f} {e['f1_macro']:>14.3f} {report['judge_mean'].get('helpfulness', 0):>11.2f}"


def main() -> None:
    root = Path(__file__).parent
    kaggle_source = root / "data" / "twcs.csv"
    build_subsample(kaggle_source if kaggle_source.exists() else None, root / "data" / "subsample.csv")
    golden = load_golden(root / "data" / "sample_golden_set.json")
    print("\nAI Customer Support Agent Benchmark")
    provider = os.getenv("LLM_PROVIDER") or ("gemini" if os.getenv("GEMINI_API_KEY") else "openai" if os.getenv("OPENAI_API_KEY") else "mock")
    print(f"Golden records: {len(golden)} | Requested provider: {provider}")
    print(f"{'System':<18} {'Intent Acc':>8} {'Intent F1':>10} {'Escalation F1':>14} {'Judge Help':>11}")
    print("-" * 66)
    headline = SupportAgent()
    for name, system in [("Majority baseline", MajorityBaseline()), ("Simple baseline", SimpleBaseline()), ("Headline pipeline", headline)]:
        print(line(name, evaluate(system, golden)))
    if headline.provider_error:
        print(f"\n{headline.provider.title()} unavailable; completed safely in mock mode ({headline.provider_error.split(':', 1)[0]}).")
    else:
        print("\nProviders: Gemini preferred when GEMINI_API_KEY is set; OpenAI and mock fallbacks remain available.")


if __name__ == "__main__":
    main()
