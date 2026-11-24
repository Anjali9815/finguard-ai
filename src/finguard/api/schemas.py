"""Request and response models for the API."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["allow", "review", "escalate"]


class InvestigationRequest(BaseModel):
    trans_num: str = Field(description="ID of the transaction to investigate.")
    force: bool = Field(default=False, description="Re-run even if this transaction was already investigated.")


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64, description="Name or ID of the human analyst.")
    final_decision: Decision
    notes: str = ""


class AuditEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    event_type: str
    actor: str
    detail: dict


class InvestigationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    trans_num: str
    created_at: datetime
    ml_risk_score: float
    anomaly_percentile: float
    amount: float
    decision: Decision
    risk_level: str
    confidence: str
    key_reasons: list[str]
    recommended_action: str
    requires_human_review: bool
    transaction_analysis: str | None = None
    account_behavior: str | None = None
    investigation_report: str | None = None
    source: str
    status: str
    human_decision: Decision | None = None
    reviewer: str | None = None
    reviewer_notes: str | None = None
    reviewed_at: datetime | None = None
