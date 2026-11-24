"""Runs the investigation for one transaction.

Two modes:
  * real  : the Google ADK multi-agent pipeline (calls Gemini, uses API quota)
  * mock  : a simple rule on the ML score, NO LLM calls. For testing the API
            without spending quota. Results are stored with source="mock".
Set FINGUARD_MOCK_AGENT=1 to use mock mode.
"""
import json
import os
import uuid

from src.finguard.agents.tools import get_card_history, get_transaction_details

APP = "finguard_api"
USER = "api"
_runner = None
_session_service = None


def mock_enabled() -> bool:
    return os.getenv("FINGUARD_MOCK_AGENT", "0") == "1"


def _get_runner():
    """Built lazily so mock mode never needs an API key."""
    global _runner, _session_service
    if _runner is None:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from finguard_agent.agent import root_agent

        _session_service = InMemorySessionService()
        _runner = Runner(agent=root_agent, app_name=APP, session_service=_session_service)
    return _runner, _session_service


def _parse(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


async def _run_real(trans_num: str) -> dict:
    from google.genai import types

    runner, svc = _get_runner()
    session_id = uuid.uuid4().hex
    await svc.create_session(app_name=APP, user_id=USER, session_id=session_id)
    msg = types.Content(role="user", parts=[types.Part(text=f"Investigate transaction {trans_num}")])
    async for _ in runner.run_async(user_id=USER, session_id=session_id, new_message=msg):
        pass
    state = (await svc.get_session(app_name=APP, user_id=USER, session_id=session_id)).state
    decision = _parse(state.get("decision"))
    if not decision.get("decision"):
        raise RuntimeError("Agent pipeline returned no decision.")
    return {
        "decision": decision,
        "transaction_analysis": state.get("transaction_analysis"),
        "account_behavior": state.get("account_behavior"),
        "investigation_report": state.get("investigation"),
        "source": "agents",
    }


def _run_mock(trans_num: str) -> dict:
    d = get_transaction_details(trans_num)
    h = get_card_history(trans_num)
    score = d["fraud_probability"]
    pct = h.get("amount_percentile_in_card_history", 0)
    if score >= 0.95 and pct >= 95:
        decision, level = "escalate", "high"
    elif score < 0.6:
        decision, level = "allow", "low"
    else:
        decision, level = "review", "medium"
    return {
        "decision": {
            "transaction_id": trans_num,
            "decision": decision,
            "risk_level": level,
            "confidence": "low",
            "key_reasons": [f"MOCK rule: risk score {score}, amount percentile {pct}"],
            "recommended_action": "Mock result for API testing. Not an agent decision.",
            "requires_human_review": decision != "allow",
        },
        "transaction_analysis": None,
        "account_behavior": None,
        "investigation_report": None,
        "source": "mock",
    }


async def run_investigation(trans_num: str) -> dict:
    if mock_enabled():
        return _run_mock(trans_num)
    return await _run_real(trans_num)
