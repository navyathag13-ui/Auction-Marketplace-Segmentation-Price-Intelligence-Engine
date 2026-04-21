"""
Pydantic response schemas
"""

from typing import Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    records_loaded: int


class SegmentInfo(BaseModel):
    segment_id: int
    segment_name: str
    listing_count: int
    avg_asking_price: float
    median_asking_price: float
    avg_age_years: Optional[float]
    avg_condition_score: Optional[float]
    avg_engagement: Optional[float]
    avg_sold_rate: Optional[float]
    avg_days_on_market: Optional[float]


class SegmentSummaryResponse(BaseModel):
    total_listings: int
    num_segments: int
    segments: list[SegmentInfo]


class ListingSegmentResponse(BaseModel):
    listing_id: str
    segment_id: Optional[int]
    segment_name: Optional[str]
    asking_price: float
    category: str
    condition: Optional[str]
    region: Optional[str]


class PriceScoreResponse(BaseModel):
    listing_id: str
    asking_price: float
    price_label: Optional[str]
    price_score: Optional[float]
    prob_underpriced: Optional[float]
    prob_fair_priced: Optional[float]
    prob_overpriced: Optional[float]
    interpretation: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


class TopFeaturesResponse(BaseModel):
    model: str
    top_features: list[FeatureImportanceItem]


class CategoryBreakdown(BaseModel):
    category: str
    count: int
    avg_price: float
    sold_rate: float


class RegionBreakdown(BaseModel):
    region: str
    count: int
    avg_price: float
    avg_engagement: float


class SegmentReportResponse(BaseModel):
    generated_at: str
    total_listings: int
    sold_rate: float
    avg_asking_price: float
    category_breakdown: list[CategoryBreakdown]
    region_breakdown: list[RegionBreakdown]
    segment_summary: list[SegmentInfo]
