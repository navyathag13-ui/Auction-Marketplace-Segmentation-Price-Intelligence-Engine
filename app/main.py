"""
Auction Marketplace Segmentation & Price Intelligence Engine
FastAPI Application Entry Point
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .db.database import init_db
from .api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB on startup."""
    log.info("Starting up — initializing database …")
    try:
        init_db()
        log.info("Database initialized.")
    except Exception as e:
        log.error(f"DB init failed: {e}")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Auction Marketplace Segmentation & Price Intelligence Engine",
    description=(
        "End-to-end analytics API for heavy equipment auction marketplace data. "
        "Provides listing segmentation, price intelligence scoring, and market insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "Auction Marketplace Segmentation & Price Intelligence Engine",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
