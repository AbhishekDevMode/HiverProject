"""Ingest and reproducibly subsample Kaggle Customer Support on Twitter threads.

Download `twcs.csv` from thoughtvector/customer-support-on-twitter and pass it to
this module.  The fallback generator keeps demos and CI runnable without Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

TARGET_BRAND = "AmazonHelp"
SEED = 7


def _synthetic_rows(count: int = 2_000) -> pd.DataFrame:
    cases = [
        ("Where is my order {n}? It was due yesterday.", "Order Tracking"),
        ("My delivery for order {n} is delayed again. This is ridiculous.", "Delivery Delay"),
        ("I was charged twice for order {n}. Please refund the duplicate.", "Refund/Billing"),
        ("I cannot sign in to my account after changing my password.", "Account Auth"),
        ("The app crashes whenever I open order {n}.", "Technical Issue"),
        ("How do I update the delivery address before an item ships?", "General Inquiry"),
        ("My package says delivered but it is not here.", "Missing Package"),
        ("I want to cancel order {n} before it ships.", "Cancellation"),
    ]
    rows = []
    for i in range(count):
        text, intent = cases[i % len(cases)]
        rows.append({
            "tweet_id": f"synthetic-{i:04d}", "author_id": f"customer-{i:04d}",
            "incoming_text": text.format(n=f"A{i:07d}"), "brand": TARGET_BRAND,
            "intent_hint": intent, "in_response_to_tweet_id": "", "historical_resolution": "",
        })
    return pd.DataFrame(rows)


def _read_brand_parents(source: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Read large Kaggle exports in bounded memory using two streaming passes."""
    parent_ids: set[str] = set()
    resolutions: dict[str, str] = {}
    columns = ["tweet_id", "author_id", "in_response_to_tweet_id", "text"]
    for chunk in pd.read_csv(source, usecols=columns, dtype=str, chunksize=chunksize, low_memory=False):
        is_brand = chunk["author_id"].fillna("").str.lstrip("@").str.casefold().eq(TARGET_BRAND.casefold())
        for parent_id, reply in chunk.loc[is_brand, ["in_response_to_tweet_id", "text"]].dropna(subset=["in_response_to_tweet_id"]).itertuples(index=False):
            parent_ids.add(str(parent_id))
            if isinstance(reply, str) and reply.strip():
                resolutions.setdefault(str(parent_id), reply.strip())
    parent_ids.discard("")
    if not parent_ids:
        raise ValueError(f"No @{TARGET_BRAND} parent interactions found in {source}")

    customer_chunks: list[pd.DataFrame] = []
    needed = ["tweet_id", "author_id", "text", "in_response_to_tweet_id"]
    for chunk in pd.read_csv(source, usecols=needed, dtype=str, chunksize=chunksize, low_memory=False):
        matched = chunk.loc[chunk["tweet_id"].isin(parent_ids)]
        if not matched.empty:
            customer_chunks.append(matched)
    if not customer_chunks:
        raise ValueError(f"No customer parent tweets found for @{TARGET_BRAND} in {source}")
    frame = pd.concat(customer_chunks, ignore_index=True).rename(columns={"text": "incoming_text"})
    frame["brand"] = TARGET_BRAND
    frame["intent_hint"] = ""
    frame["historical_resolution"] = frame["tweet_id"].map(resolutions).fillna("")
    return frame[["tweet_id", "author_id", "incoming_text", "brand", "intent_hint", "in_response_to_tweet_id", "historical_resolution"]]


def build_subsample(source: Path | None, output: Path, n: int = 2_000) -> pd.DataFrame:
    """Save a stable, de-duplicated brand subset. Supports Kaggle's `twcs.csv`."""
    if source and source.exists():
        frame = _read_brand_parents(source)
        if frame.empty:
            raise ValueError(f"No @{TARGET_BRAND} parent interactions found in {source}")
    else:
        frame = _synthetic_rows(max(n, 200))
    frame = frame.dropna(subset=["incoming_text"]).drop_duplicates("tweet_id")
    frame = frame.assign(_stable=frame.tweet_id.astype(str).map(lambda x: hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest()))
    result = frame.sort_values("_stable").head(n).drop(columns="_stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Path to Kaggle twcs.csv")
    parser.add_argument("--output", type=Path, default=Path("data/subsample.csv"))
    parser.add_argument("--n", type=int, default=2_000)
    args = parser.parse_args()
    print(f"Wrote {len(build_subsample(args.source, args.output, args.n))} rows to {args.output}")
