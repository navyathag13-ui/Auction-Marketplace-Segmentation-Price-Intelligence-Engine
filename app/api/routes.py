"""
FastAPI route handlers
"""

import os
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, Float, cast

from ..db.database import get_db
from ..db.models import (
    Listing, ListingSegment, ListingPrediction,
    ListingFeatures, SegmentSummary, Seller
)
from ..models.schemas import (
    HealthResponse, SegmentSummaryResponse, SegmentInfo,
    ListingSegmentResponse, PriceScoreResponse,
    TopFeaturesResponse, FeatureImportanceItem,
    SegmentReportResponse, CategoryBreakdown, RegionBreakdown,
)
from ..services.model_service import load_feature_importances

router = APIRouter()

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """API health check and metadata."""
    try:
        count = db.query(func.count(Listing.listing_id)).scalar()
        db_status = "connected"
    except Exception:
        count = 0
        db_status = "error"

    return HealthResponse(
        status="ok",
        version="1.0.0",
        database=db_status,
        records_loaded=count or 0,
    )


# ── Segments ──────────────────────────────────────────────────────────────────

@router.get("/segments/summary", response_model=SegmentSummaryResponse, tags=["Segmentation"])
def get_segments_summary(db: Session = Depends(get_db)):
    """Return high-level profile of each discovered segment."""
    summaries = db.query(SegmentSummary).order_by(SegmentSummary.segment_id).all()
    if not summaries:
        raise HTTPException(status_code=404, detail="No segment summaries found. Run init_db.py first.")

    total = db.query(func.count(Listing.listing_id)).scalar() or 0

    return SegmentSummaryResponse(
        total_listings=total,
        num_segments=len(summaries),
        segments=[
            SegmentInfo(
                segment_id=s.segment_id,
                segment_name=s.segment_name,
                listing_count=s.listing_count,
                avg_asking_price=round(s.avg_asking_price, 2),
                median_asking_price=round(s.median_asking_price, 2),
                avg_age_years=round(s.avg_age_years, 2) if s.avg_age_years else None,
                avg_condition_score=round(s.avg_condition_score, 3) if s.avg_condition_score else None,
                avg_engagement=round(s.avg_engagement, 2) if s.avg_engagement else None,
                avg_sold_rate=round(s.avg_sold_rate, 4) if s.avg_sold_rate else None,
                avg_days_on_market=round(s.avg_days_on_market, 2) if s.avg_days_on_market else None,
            )
            for s in summaries
        ],
    )


@router.get("/listing/{listing_id}/segment", response_model=ListingSegmentResponse, tags=["Segmentation"])
def get_listing_segment(listing_id: str, db: Session = Depends(get_db)):
    """Return segment assignment for a specific listing."""
    listing = db.query(Listing).filter(Listing.listing_id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")

    seg = db.query(ListingSegment).filter(ListingSegment.listing_id == listing_id).first()

    return ListingSegmentResponse(
        listing_id=listing.listing_id,
        segment_id=seg.segment_kmeans if seg else None,
        segment_name=seg.segment_label if seg else None,
        asking_price=listing.asking_price,
        category=listing.category,
        condition=listing.condition,
        region=listing.region,
    )


# ── Price Intelligence ─────────────────────────────────────────────────────────

@router.get("/listing/{listing_id}/price-score", response_model=PriceScoreResponse, tags=["Pricing"])
def get_price_score(listing_id: str, db: Session = Depends(get_db)):
    """Return price intelligence score for a listing."""
    listing = db.query(Listing).filter(Listing.listing_id == listing_id).first()
    if not listing:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found")

    pred = db.query(ListingPrediction).filter(
        ListingPrediction.listing_id == listing_id
    ).first()

    if pred:
        interpretation_map = {
            "underpriced":  "This listing appears underpriced relative to comparable equipment. Strong buy signal.",
            "fair_priced":  "This listing is priced in line with market comparables.",
            "overpriced":   "This listing appears overpriced relative to comparable equipment. Negotiate or wait.",
        }
        interpretation = interpretation_map.get(pred.price_label_pred, "No interpretation available.")
        return PriceScoreResponse(
            listing_id=listing_id,
            asking_price=listing.asking_price,
            price_label=pred.price_label_pred,
            price_score=pred.price_score,
            prob_underpriced=pred.prob_underpriced,
            prob_fair_priced=pred.prob_fair_priced,
            prob_overpriced=pred.prob_overpriced,
            interpretation=interpretation,
        )

    # Fallback: use features table label
    feat = db.query(ListingFeatures).filter(ListingFeatures.listing_id == listing_id).first()
    label = feat.price_label if feat else "fair_priced"
    return PriceScoreResponse(
        listing_id=listing_id,
        asking_price=listing.asking_price,
        price_label=label,
        price_score=None,
        prob_underpriced=None,
        prob_fair_priced=None,
        prob_overpriced=None,
        interpretation="Approximate label from rule-based heuristic (model probabilities unavailable).",
    )


# ── Insights ───────────────────────────────────────────────────────────────────

@router.get("/insights/top-features", response_model=TopFeaturesResponse, tags=["Insights"])
def get_top_features(top_n: int = 15):
    """Return top N most important features from the trained LightGBM model."""
    fi = load_feature_importances()
    if not fi:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    top = fi[:top_n]
    return TopFeaturesResponse(
        model="LightGBM Price Classifier",
        top_features=[
            FeatureImportanceItem(feature=f["feature"], importance=f["importance"], rank=f["rank"])
            for f in top
        ],
    )


# ── Reports ────────────────────────────────────────────────────────────────────

@router.get("/reports/segment-summary", response_model=SegmentReportResponse, tags=["Reports"])
def get_segment_report(db: Session = Depends(get_db)):
    """Full segment report with category and region breakdowns."""
    total = db.query(func.count(Listing.listing_id)).scalar() or 0
    sold_count = db.query(func.count(Listing.listing_id)).filter(Listing.sold == True).scalar() or 0
    avg_price = db.query(func.avg(Listing.asking_price)).scalar() or 0.0

    # Category breakdown
    cat_rows = (
        db.query(
            Listing.category,
            func.count(Listing.listing_id).label("count"),
            func.avg(Listing.asking_price).label("avg_price"),
            func.avg(cast(Listing.sold, Float)).label("sold_rate"),
        )
        .group_by(Listing.category)
        .order_by(func.count(Listing.listing_id).desc())
        .all()
    )

    # Region breakdown
    reg_rows = (
        db.query(
            Listing.region,
            func.count(Listing.listing_id).label("count"),
            func.avg(Listing.asking_price).label("avg_price"),
            func.avg(Listing.views).label("avg_views"),
        )
        .group_by(Listing.region)
        .order_by(func.count(Listing.listing_id).desc())
        .all()
    )

    # Segment summaries
    summaries = db.query(SegmentSummary).order_by(SegmentSummary.segment_id).all()

    return SegmentReportResponse(
        generated_at=datetime.utcnow().isoformat(),
        total_listings=total,
        sold_rate=round(sold_count / total, 4) if total else 0,
        avg_asking_price=round(float(avg_price), 2),
        category_breakdown=[
            CategoryBreakdown(
                category=r.category,
                count=r.count,
                avg_price=round(float(r.avg_price or 0), 2),
                sold_rate=round(float(r.sold_rate or 0), 4),
            )
            for r in cat_rows
        ],
        region_breakdown=[
            RegionBreakdown(
                region=r.region or "Unknown",
                count=r.count,
                avg_price=round(float(r.avg_price or 0), 2),
                avg_engagement=round(float(r.avg_views or 0), 2),
            )
            for r in reg_rows
        ],
        segment_summary=[
            SegmentInfo(
                segment_id=s.segment_id,
                segment_name=s.segment_name,
                listing_count=s.listing_count,
                avg_asking_price=round(s.avg_asking_price, 2),
                median_asking_price=round(s.median_asking_price, 2),
                avg_age_years=round(s.avg_age_years, 2) if s.avg_age_years else None,
                avg_condition_score=round(s.avg_condition_score, 3) if s.avg_condition_score else None,
                avg_engagement=round(s.avg_engagement, 2) if s.avg_engagement else None,
                avg_sold_rate=round(s.avg_sold_rate, 4) if s.avg_sold_rate else None,
                avg_days_on_market=round(s.avg_days_on_market, 2) if s.avg_days_on_market else None,
            )
            for s in summaries
        ],
    )
