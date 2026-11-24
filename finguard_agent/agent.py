"""FinGuard AI: multi-agent fraud investigation pipeline (Google ADK).

Flow (SequentialAgent, each agent writes its result into shared session state):

    transaction_analyst  -> state["transaction_analysis"]
    account_behavior     -> state["account_behavior"]
    fraud_investigator   -> state["investigation"]
    decision_agent       -> state["decision"]   (structured JSON)

Run:   adk web        (from the project root, then pick "finguard_agent")
Ask:   "Investigate transaction <trans_num>"
"""
import os
from typing import Literal

from pydantic import BaseModel, Field
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

from src.finguard.agents.tools import (
    explain_risk_factors,
    get_card_history,
    get_transaction_details,
    list_flagged_transactions,
)

# Retry automatically on temporary Google errors (503 overloaded, 429 rate limit).
MODEL = Gemini(
    model=os.getenv("FINGUARD_MODEL", "gemini-flash-latest"),
    retry_options=types.HttpRetryOptions(
        attempts=6,
        initial_delay=2.0,
        max_delay=60.0,
        exp_base=2.0,
        http_status_codes=[429, 500, 503, 504],
    ),
)

# Shared wording rule: the ML score is NOT a literal probability because the
# model was trained with a heavy fraud class weight (scale_pos_weight ~170).
SCORE_NOTE = (
    "Always call the ML output a 'risk score', never a literal probability: "
    "the model is trained with a heavy fraud class weight, so scores run high. "
    "Never invent facts. Use only what the tools and earlier agents provided. "
    "Never cite 'the score exceeds the threshold' as evidence: the threshold only "
    "explains why the case was flagged, it says nothing about whether it is fraud."
)

# --------------------------------------------------------------- Agent 1
transaction_analyst = Agent(
    name="transaction_analyst",
    model=MODEL,
    description="Examines the flagged transaction and the ML model's reasons.",
    instruction=f"""You are a fraud analyst examining one banking transaction.
The user's message contains a transaction ID. Take it from there.

1. Call get_transaction_details for the ID.
2. Call explain_risk_factors for the same ID.
3. Write a short factual summary: amount, merchant, category, time, location,
   the risk score, whether it was flagged, the anomaly percentile, and the top
   factors in plain English (positive contribution = raises risk).
If a tool returns status 'error', say so clearly and stop.
{SCORE_NOTE}""",
    tools=[get_transaction_details, explain_risk_factors],
    output_key="transaction_analysis",
)

# --------------------------------------------------------------- Agent 2
account_behavior = Agent(
    name="account_behavior_analyst",
    model=MODEL,
    description="Compares the transaction with the card's own history.",
    instruction=f"""You analyse the cardholder's historical behaviour.

Transaction analysis so far:
{{transaction_analysis}}

Find the transaction ID in the analysis above and call get_card_history with it.
Then state whether this transaction fits the card's normal pattern:
- Amount: use amount_percentile_in_card_history and current_amount_vs_avg.
- Timing: use share_of_past_near_this_hour and share_of_past_at_night. Only call
  the hour unusual if fewer than about 5% of past transactions were near it.
  A late hour is NOT unusual for a card that often transacts then.
- Novelty: category_seen_before and merchant_seen_before.
- Velocity: transactions_last_1h and transactions_last_24h.
IMPORTANT: if the card has fewer than about 10 prior transactions, say the
history is thin and your conclusions are low-confidence.
{SCORE_NOTE}""",
    tools=[get_card_history],
    output_key="account_behavior",
)

# --------------------------------------------------------------- Agent 3
fraud_investigator = Agent(
    name="fraud_investigator",
    model=MODEL,
    description="Correlates all evidence and explains the risk.",
    instruction=f"""You are a senior fraud investigator. Combine the evidence.

Transaction analysis:
{{transaction_analysis}}

Account behaviour:
{{account_behavior}}

Write a concise investigation report: (1) signals that support fraud,
(2) signals that argue it is legitimate, (3) data gaps or caveats, and
(4) an overall assessment in plain language a bank analyst can act on.

Weigh the evidence like this:
- STRONG fraud evidence: very high risk score together with a high anomaly
  percentile; an amount far above the card's norm (high percentile in its own
  history, many multiples of its average); activity at an hour the card rarely
  uses; a burst of transactions within an hour.
- STRONG legitimacy evidence: amount, hour and velocity all typical for this card
  and a low anomaly percentile.
- WEAK legitimacy evidence: a familiar merchant or category, or a long history on
  its own. Fraudsters often use common merchants and categories, so weak evidence
  must not override strong fraud evidence.
Be balanced, and state clearly when strong signals conflict.
{SCORE_NOTE}""",
    output_key="investigation",
)


# --------------------------------------------------------------- Agent 4
class Decision(BaseModel):
    transaction_id: str
    decision: Literal["allow", "review", "escalate"] = Field(
        description="allow = let it through; review = send to a human analyst; "
        "escalate = urgent, hold the transaction and contact the customer."
    )
    risk_level: Literal["low", "medium", "high"]
    confidence: Literal["low", "medium", "high"]
    key_reasons: list[str] = Field(description="2 to 4 short reasons.")
    recommended_action: str
    requires_human_review: bool


decision_agent = Agent(
    name="decision_agent",
    model=MODEL,
    description="Recommends allow, review or escalate.",
    instruction=f"""You make the final recommendation.

Investigation report:
{{investigation}}

Choose 'allow', 'review' or 'escalate'. Rules of thumb: strong fraud signals that
agree across the ML output AND the card history -> escalate; mixed, conflicting
or thin evidence -> review; a clear legitimate pattern (typical amount, hour and
velocity, modest anomaly percentile) -> allow. Do not choose 'review' just
because the case was flagged: if the card's own behaviour clearly explains the
transaction, allow it. A human always reviews 'review' and 'escalate'.
Set confidence to low when card history is thin or evidence conflicts.
{SCORE_NOTE}""",
    output_schema=Decision,
    output_key="decision",
)

# ------------------------------------------------------------- Pipeline
root_agent = SequentialAgent(
    name="finguard_pipeline",
    description="Investigates a flagged banking transaction end to end.",
    sub_agents=[transaction_analyst, account_behavior, fraud_investigator, decision_agent],
)
