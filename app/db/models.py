"""
SQLAlchemy ORM Models
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime,
    Text, ForeignKey, Index, create_engine
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Seller(Base):
    __tablename__ = "sellers"

    seller_id        = Column(String(10), primary_key=True)
    seller_type      = Column(String(30), nullable=False)
    tenure_years     = Column(Float)
    total_listings   = Column(Integer)
    avg_rating       = Column(Float)
    verified         = Column(Boolean, default=False)

    listings = relationship("Listing", back_populates="seller")


class Listing(Base):
    __tablename__ = "listings"

    listing_id       = Column(String(10), primary_key=True)
    seller_id        = Column(String(10), ForeignKey("sellers.seller_id"), nullable=False)
    category         = Column(String(50), nullable=False)
    make             = Column(String(50))
    year             = Column(Integer)
    condition        = Column(String(20))
    hours_used       = Column(Float)
    asking_price     = Column(Float, nullable=False)
    region           = Column(String(30))
    list_date        = Column(DateTime)
    views            = Column(Integer, default=0)
    bids             = Column(Integer, default=0)
    inquiries        = Column(Integer, default=0)
    sold             = Column(Boolean, default=False)
    days_on_market   = Column(Integer)
    final_price      = Column(Float)
    age_years        = Column(Float)
    price_band       = Column(String(20))
    list_year        = Column(Integer)
    list_month       = Column(Integer)

    seller = relationship("Seller", back_populates="listings")
    features = relationship("ListingFeatures", back_populates="listing", uselist=False)
    segment  = relationship("ListingSegment",  back_populates="listing", uselist=False)
    prediction = relationship("ListingPrediction", back_populates="listing", uselist=False)

    __table_args__ = (
        Index("idx_listings_category", "category"),
        Index("idx_listings_region", "region"),
        Index("idx_listings_seller", "seller_id"),
    )


class ListingFeatures(Base):
    __tablename__ = "listing_features"

    id                       = Column(Integer, primary_key=True, autoincrement=True)
    listing_id               = Column(String(10), ForeignKey("listings.listing_id"), unique=True)
    price_vs_cat_median      = Column(Float)
    price_vs_cond_median     = Column(Float)
    price_zscore_cat         = Column(Float)
    log_asking_price         = Column(Float)
    hours_per_year           = Column(Float)
    hours_vs_cat_median      = Column(Float)
    total_engagement         = Column(Integer)
    bid_rate                 = Column(Float)
    inquiry_rate             = Column(Float)
    condition_score          = Column(Integer)
    seller_listing_count     = Column(Integer)
    seller_sold_rate         = Column(Float)
    category_freq            = Column(Float)
    region_demand_proxy      = Column(Float)
    price_label              = Column(String(20))

    listing = relationship("Listing", back_populates="features")


class ListingSegment(Base):
    __tablename__ = "listing_segments"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    listing_id     = Column(String(10), ForeignKey("listings.listing_id"), unique=True)
    segment_kmeans = Column(Integer)
    segment_label  = Column(String(50))
    assigned_at    = Column(DateTime, default=datetime.utcnow)

    listing = relationship("Listing", back_populates="segment")

    __table_args__ = (
        Index("idx_segments_label", "segment_label"),
    )


class ListingPrediction(Base):
    __tablename__ = "listing_predictions"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    listing_id           = Column(String(10), ForeignKey("listings.listing_id"), unique=True)
    price_label_pred     = Column(String(20))
    price_label_enc_pred = Column(Integer)
    prob_underpriced     = Column(Float)
    prob_fair_priced     = Column(Float)
    prob_overpriced      = Column(Float)
    price_score          = Column(Float)  # 0–100 composite
    predicted_at         = Column(DateTime, default=datetime.utcnow)

    listing = relationship("Listing", back_populates="prediction")


class SegmentSummary(Base):
    __tablename__ = "segment_summaries"

    segment_id          = Column(Integer, primary_key=True)
    segment_name        = Column(String(50))
    listing_count       = Column(Integer)
    avg_asking_price    = Column(Float)
    median_asking_price = Column(Float)
    avg_age_years       = Column(Float)
    avg_condition_score = Column(Float)
    avg_engagement      = Column(Float)
    avg_sold_rate       = Column(Float)
    avg_days_on_market  = Column(Float)
    updated_at          = Column(DateTime, default=datetime.utcnow)
