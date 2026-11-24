"""Evaluate the FinGuard agent pipeline against ground truth.

Samples flagged transactions (half real fraud, half false alarms), runs the
agent pipeline on each, and measures what the agents decided.

Run from the project root:
    python -m src.finguard.eval_agents --per-class 10
    python -m src.finguard.eval_agents --per-class 10 --dry-run   # no API calls
"""
import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.finguard.tools import ROOT, STORE, _config

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "finguard_agent" / ".env")

if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    raise SystemExit("No API key found. Put GOOGLE_API_KEY=... in .env")

APP = "finguard_eval"
USER = "evaluator"
OUT = ROOT / "reports" / "agent_eval.csv"


def sample_cases(per_class: int, seed: int = 42) -> pd.DataFrame:
    """Flagged test transactions: per_class real frauds + per_class false alarms."""
    df = pd.read_parquet(STORE, columns=["trans_num", "split", "is_fraud", "fraud_prob", "amt", "category"])
    flagged = df[(df.split == "test") & (df.fraud_prob >= _config()["threshold"])]
    fraud = flagged[flagged.is_fraud == 1].sample(per_class, random_state=seed)
    legit = flagged[flagged.is_fraud == 0].sample(per_class, random_state=seed)
    return pd.concat([fraud, legit]).sample(frac=1, random_state=seed).reset_index(drop=True)


def parse_decision(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {}


async def run_one(runner, session_service, trans_num: str) -> dict:
    from google.genai import types

    session_id = uuid.uuid4().hex
    await session_service.create_session(app_name=APP, user_id=USER, session_id=session_id)
    msg = types.Content(role="user", parts=[types.Part(text=f"Investigate transaction {trans_num}")])
    async for _ in runner.run_async(user_id=USER, session_id=session_id, new_message=msg):
        pass
    session = await session_service.get_session(app_name=APP, user_id=USER, session_id=session_id)
    return parse_decision(session.state.get("decision"))


async def main_async(cases: pd.DataFrame, concurrency: int):
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from finguard_agent.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP, session_service=session_service)
    sem = asyncio.Semaphore(concurrency)
    rows = []

    async def worker(i, r):
        async with sem:
            try:
                d = await run_one(runner, session_service, r.trans_num)
                err = ""
            except Exception as e:  # keep going if one case fails
                d, err = {}, f"{type(e).__name__}: {e}"[:200]
            rows.append({**r._asdict(), "decision": d.get("decision"), "risk_level": d.get("risk_level"),
                         "confidence": d.get("confidence"), "reasons": " | ".join(d.get("key_reasons", [])),
                         "error": err})
            print(f"[{len(rows)}/{len(cases)}] {r.trans_num[:8]} truth={r.is_fraud} -> {d.get('decision')} {err}")

    await asyncio.gather(*(worker(i, r) for i, r in enumerate(cases.itertuples(index=False))))
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame):
    ok = res[res.decision.notna()]
    print(f"\nCompleted {len(ok)} of {len(res)} cases\n")
    print(pd.crosstab(ok.is_fraud.map({1: "REAL FRAUD", 0: "FALSE ALARM"}), ok.decision, margins=True))
    fraud, legit = ok[ok.is_fraud == 1], ok[ok.is_fraud == 0]
    if len(fraud):
        print(f"\nReal fraud sent to a human (review/escalate): {fraud.decision.isin(['review', 'escalate']).mean():.0%}")
        print(f"Real fraud wrongly ALLOWED                   : {(fraud.decision == 'allow').mean():.0%}")
    if len(legit):
        print(f"False alarms cleared automatically (allow)   : {(legit.decision == 'allow').mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed: tune prompts on one seed, report results on a different one")
    ap.add_argument("--out", default=None, help="output csv name inside reports/")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.out:
        OUT = ROOT / "reports" / a.out

    cases = sample_cases(a.per_class, seed=a.seed)
    if a.dry_run:
        print(cases.head(10).to_string())
        print("\nclass counts:\n", cases.is_fraud.value_counts())
    else:
        res = asyncio.run(main_async(cases, a.concurrency))
        OUT.parent.mkdir(exist_ok=True)
        res.to_csv(OUT, index=False)
        summarise(res)
        print(f"\nsaved {OUT}")