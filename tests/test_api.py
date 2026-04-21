"""
FastAPI endpoint tests
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "database" in data
        assert "records_loaded" in data

    def test_health_status_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json()["status"] == "ok"


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "docs" in data


class TestSegmentsEndpoint:
    def test_segments_summary_structure(self, client):
        resp = client.get("/api/v1/segments/summary")
        if resp.status_code == 404:
            pytest.skip("Segments not initialized — run init_db.py first")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_listings" in data
        assert "num_segments" in data
        assert "segments" in data
        assert isinstance(data["segments"], list)

    def test_segment_info_fields(self, client):
        resp = client.get("/api/v1/segments/summary")
        if resp.status_code == 404:
            pytest.skip("Segments not initialized")
        data = resp.json()
        if data["segments"]:
            seg = data["segments"][0]
            assert "segment_id" in seg
            assert "segment_name" in seg
            assert "listing_count" in seg
            assert "avg_asking_price" in seg


class TestListingEndpoints:
    @pytest.fixture(autouse=True)
    def check_db_populated(self, client):
        resp = client.get("/api/v1/health")
        if resp.json().get("records_loaded", 0) == 0:
            pytest.skip("Database not populated — run init_db.py first")

    def test_invalid_listing_returns_404(self, client):
        resp = client.get("/api/v1/listing/NONEXISTENT/segment")
        assert resp.status_code == 404

    def test_invalid_listing_price_score_404(self, client):
        resp = client.get("/api/v1/listing/NONEXISTENT/price-score")
        assert resp.status_code == 404

    def test_valid_listing_segment(self, client):
        # Use first listing from DB via health check
        resp = client.get("/api/v1/listing/L000000/segment")
        if resp.status_code == 404:
            pytest.skip("Listing L000000 not in DB")
        assert resp.status_code == 200
        data = resp.json()
        assert "listing_id" in data
        assert "asking_price" in data
        assert "category" in data

    def test_valid_listing_price_score(self, client):
        resp = client.get("/api/v1/listing/L000000/price-score")
        if resp.status_code == 404:
            pytest.skip("Listing L000000 not in DB")
        assert resp.status_code == 200
        data = resp.json()
        assert "price_label" in data
        assert "interpretation" in data


class TestInsightsEndpoints:
    def test_top_features_with_model(self, client):
        resp = client.get("/api/v1/insights/top-features")
        if resp.status_code == 503:
            pytest.skip("Model not loaded — run train_model.py first")
        assert resp.status_code == 200
        data = resp.json()
        assert "top_features" in data
        assert len(data["top_features"]) > 0

    def test_segment_report(self, client):
        resp = client.get("/api/v1/reports/segment-summary")
        if resp.status_code == 404:
            pytest.skip("DB not populated")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_listings" in data
        assert "category_breakdown" in data
        assert "region_breakdown" in data
