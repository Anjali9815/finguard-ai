"""FinGuard AI REST API.

Run from the project root:
    uvicorn src.finguard.api.main:app --reload
    (add FINGUARD_MOCK_AGENT=1 in front to test without calling Gemini)
Interactive docs:  http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.finguard.agents.tools import (
    explain_risk_factors,
    get_card_history,
    get_transaction_details,
    list_flagged_transactions,
)

from src.finguard.api import agent_runner
from src.finguard.api.db import AuditEvent, Investigation, get_db, init_db, utcnow
from src.finguard.api.schemas import (
    AuditEventOut,
    InvestigationOut,
    InvestigationRequest,
    ReviewRequest,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="FinGuard AI",
    description="Agentic fraud investigation: ML scoring, multi-agent analysis, human review, audit trail.",
    version="0.1.0",
    lifespan=lifespan,
)


def _audit(db: Session, inv: Investigation, event_type: str, actor: str, detail: dict) -> None:
    db.add(AuditEvent(investigation_id=inv.id, event_type=event_type, actor=actor, detail=detail))


@app.get("/health")
def health():
    return {"status": "ok", "mock_agent": agent_runner.mock_enabled()}


@app.get("/queue")
def queue(limit: int = Query(10, ge=1, le=20)):
    """Transactions the ML model flagged, highest risk first."""
    return list_flagged_transactions(limit)


@app.get("/transactions/{trans_num}/evidence")
def evidence(trans_num: str):
    """Raw evidence for one transaction (ML score, risk factors, card history). No LLM involved."""
    details = get_transaction_details(trans_num)
    if details["status"] == "error":
        raise HTTPException(404, details["message"])
    return {
        "details": details,
        "risk_factors": explain_risk_factors(trans_num),
        "card_history": get_card_history(trans_num),
    }


@app.post("/investigations", response_model=InvestigationOut)
async def investigate(req: InvestigationRequest, db: Session = Depends(get_db)):
    """Run the agent pipeline on a transaction and store the result.

    If the transaction was already investigated, the stored result is returned
    (saves API quota). Pass force=true to run it again.
    """
    details = get_transaction_details(req.trans_num)
    if details["status"] == "error":
        raise HTTPException(404, details["message"])

    if not req.force:
        existing = db.scalar(
            select(Investigation).where(Investigation.trans_num == req.trans_num).order_by(Investigation.id.desc())
        )
        if existing:
            return existing

    try:
        result = await agent_runner.run_investigation(req.trans_num)
    except Exception as e:  # e.g. rate limit, model unavailable
        raise HTTPException(502, f"Agent pipeline failed: {type(e).__name__}: {str(e)[:200]}")

    d = result["decision"]
    inv = Investigation(
        trans_num=req.trans_num,
        ml_risk_score=details["fraud_probability"],
        anomaly_percentile=details["anomaly_percentile"],
        amount=details["amount"],
        decision=d["decision"],
        risk_level=d.get("risk_level", "medium"),
        confidence=d.get("confidence", "low"),
        key_reasons=d.get("key_reasons", []),
        recommended_action=d.get("recommended_action", ""),
        requires_human_review=bool(d.get("requires_human_review", d["decision"] != "allow")),
        transaction_analysis=result["transaction_analysis"],
        account_behavior=result["account_behavior"],
        investigation_report=result["investigation_report"],
        source=result["source"],
        status="auto_cleared" if d["decision"] == "allow" and not d.get("requires_human_review") else "pending_review",
    )
    db.add(inv)
    db.flush()  # assigns inv.id
    _audit(db, inv, "investigation_created", "agent_pipeline",
           {"decision": inv.decision, "source": inv.source, "ml_risk_score": inv.ml_risk_score})
    db.commit()
    return inv


@app.get("/investigations", response_model=list[InvestigationOut])
def list_investigations(
    status: str | None = None,
    decision: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    q = select(Investigation).order_by(Investigation.id.desc()).limit(limit)
    if status:
        q = q.where(Investigation.status == status)
    if decision:
        q = q.where(Investigation.decision == decision)
    return db.scalars(q).all()


def _get_or_404(db: Session, investigation_id: int) -> Investigation:
    inv = db.get(Investigation, investigation_id)
    if inv is None:
        raise HTTPException(404, f"Investigation {investigation_id} not found.")
    return inv


@app.get("/investigations/{investigation_id}", response_model=InvestigationOut)
def get_investigation(investigation_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, investigation_id)


@app.post("/investigations/{investigation_id}/review", response_model=InvestigationOut)
def review(investigation_id: int, req: ReviewRequest, db: Session = Depends(get_db)):
    """A human analyst confirms or overrides the agents' recommendation."""
    inv = _get_or_404(db, investigation_id)
    if inv.status == "reviewed":
        raise HTTPException(409, "This investigation has already been reviewed.")
    inv.human_decision = req.final_decision
    inv.reviewer = req.reviewer
    inv.reviewer_notes = req.notes
    inv.reviewed_at = utcnow()
    inv.status = "reviewed"
    _audit(db, inv, "human_review", req.reviewer, {
        "agent_decision": inv.decision,
        "final_decision": req.final_decision,
        "overridden": req.final_decision != inv.decision,
        "notes": req.notes,
    })
    db.commit()
    return inv


@app.get("/investigations/{investigation_id}/audit", response_model=list[AuditEventOut])
def audit_trail(investigation_id: int, db: Session = Depends(get_db)):
    return _get_or_404(db, investigation_id).audit_events


@app.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    """Operational summary for the monitoring dashboard."""
    total = db.scalar(select(func.count(Investigation.id))) or 0
    by_decision = dict(db.execute(select(Investigation.decision, func.count()).group_by(Investigation.decision)).all())
    by_status = dict(db.execute(select(Investigation.status, func.count()).group_by(Investigation.status)).all())
    reviewed = db.scalars(select(Investigation).where(Investigation.status == "reviewed")).all()
    overrides = sum(1 for i in reviewed if i.human_decision != i.decision)
    return {
        "total_investigations": total,
        "by_agent_decision": by_decision,
        "by_status": by_status,
        "pending_human_review": by_status.get("pending_review", 0),
        "human_reviews": len(reviewed),
        "human_override_rate": round(overrides / len(reviewed), 3) if reviewed else None,
    }
