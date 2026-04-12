"""Tests for airline sentiment analysis pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.data import clean_text, assign_category, assign_criticality, prepare_dataset


class TestCleanText:
    def test_removes_mentions(self):
        assert "@user" not in clean_text("Hello @user how are you")
    
    def test_removes_urls(self):
        assert "http" not in clean_text("Check http://example.com")
    
    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"
    
    def test_preserves_content(self):
        result = clean_text("Flight was delayed 3 hours!")
        assert "delayed" in result
        assert "3" in result
    
    def test_handles_negation(self):
        result = clean_text("There were no delays at all")
        assert "no_delays" in result
    
    def test_handles_empty(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestCategoryAssignment:
    def test_baggage(self):
        assert assign_category("My luggage was lost") == "baggage"
    
    def test_delay(self):
        assert assign_category("Flight delayed by 4 hours") == "delay"
    
    def test_checkin(self):
        assert assign_category("Check-in queue was too long") == "check-in"
    
    def test_inflight(self):
        assert assign_category("The seat was uncomfortable and food terrible") == "in-flight"
    
    def test_service(self):
        assert assign_category("Staff was very rude to me") == "customer_service"
    
    def test_other(self):
        assert assign_category("xyz abc 123") == "other"


class TestCriticality:
    def test_high_negative(self):
        assert assign_criticality("negative", "This is the worst terrible unacceptable service") == "high"
    
    def test_medium_negative(self):
        assert assign_criticality("negative", "Not happy with the delay") == "medium"
    
    def test_low_positive(self):
        assert assign_criticality("positive", "Great flight experience") == "low"
    
    def test_low_neutral(self):
        assert assign_criticality("neutral", "Average experience") == "low"


class TestDataPipeline:
    def test_prepare_dataset_runs(self):
        df = prepare_dataset()
        assert len(df) > 100
        assert "text" in df.columns
        assert "sentiment_id" in df.columns
        assert "category_id" in df.columns
        assert "criticality_id" in df.columns
    
    def test_no_empty_texts(self):
        df = prepare_dataset()
        assert (df["text"].str.len() > 5).all()
    
    def test_valid_label_ids(self):
        df = prepare_dataset()
        assert df["sentiment_id"].isin([0, 1, 2]).all()
        assert df["category_id"].between(0, 6).all()
        assert df["criticality_id"].isin([0, 1, 2]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
