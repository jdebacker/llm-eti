"""Test cache exploration utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_eti.cache_utils import CacheExplorer, compare_cached_vs_fresh
from llm_eti.edsl_client import EDSLClient


class TestCacheExplorer:
    """Test the cache exploration utilities."""
    
    @patch("llm_eti.cache_utils.Cache")
    def test_cache_stats(self, mock_cache_class):
        """Test getting cache statistics."""
        # Mock cache with sample entries
        mock_cache = MagicMock()
        mock_cache.all.return_value = [
            {
                "model": "gpt-4o-mini",
                "question": {"type": "numerical", "text": "What is your taxable income?"},
                "answer": {"taxable_income": 70000},
            },
            {
                "model": "gpt-4o-mini",
                "question": {"type": "numerical", "text": "What is your taxable income?"},
                "answer": {"taxable_income": 72000},
            },
            {
                "model": "gpt-4o",
                "question": {"type": "numerical", "text": "What is your taxable income?"},
                "answer": {"taxable_income": 71000},
            },
        ]
        mock_cache_class.return_value = mock_cache
        
        explorer = CacheExplorer()
        stats = explorer.get_cache_stats()
        
        assert stats["total_entries"] == 3
        assert stats["models"]["gpt-4o-mini"] == 2
        assert stats["models"]["gpt-4o"] == 1
        assert stats["questions"]["numerical"] == 3
    
    @patch("llm_eti.cache_utils.Cache")
    def test_find_tax_scenarios(self, mock_cache_class):
        """Test finding tax scenarios in cache."""
        mock_cache = MagicMock()
        mock_cache.all.return_value = [
            {
                "model": "gpt-4o-mini",
                "question": {"text": "Your taxable income last year was $75,000"},
                "answer": {"taxable_income": 70000},
                "created_at": "2024-01-01",
            },
            {
                "model": "gpt-4o",
                "question": {"text": "What is your favorite color?"},
                "answer": {"color": "blue"},
            },
        ]
        mock_cache_class.return_value = mock_cache
        
        explorer = CacheExplorer()
        
        # Find all tax scenarios
        scenarios = explorer.find_tax_scenarios()
        assert len(scenarios) == 1
        assert scenarios[0]["model"] == "gpt-4o-mini"
        
        # Filter by model
        scenarios_mini = explorer.find_tax_scenarios(model="gpt-4o-mini")
        assert len(scenarios_mini) == 1
        
        scenarios_4o = explorer.find_tax_scenarios(model="gpt-4o")
        assert len(scenarios_4o) == 0
    
    @patch("llm_eti.cache_utils.Cache")
    def test_cost_savings_estimation(self, mock_cache_class):
        """Test cost savings estimation."""
        mock_cache = MagicMock()
        mock_cache.all.return_value = [{"entry": i} for i in range(100)]
        mock_cache_class.return_value = mock_cache
        
        explorer = CacheExplorer()
        savings = explorer.estimate_cost_savings(cost_per_1k_tokens=0.0002)
        
        assert savings["cached_responses"] == 100
        assert savings["estimated_tokens_saved"] == 10000  # 100 * 100 tokens
        assert savings["estimated_cost_saved"] == pytest.approx(0.002)  # $0.002
    
    @patch("llm_eti.cache_utils.pd.DataFrame.to_csv")
    @patch("llm_eti.cache_utils.Cache")
    def test_export_cache_data(self, mock_cache_class, mock_to_csv):
        """Test exporting cache data."""
        mock_cache = MagicMock()
        mock_cache.all.return_value = [
            {
                "model": "gpt-4o-mini",
                "question": {"text": "Your taxable income..."},
                "answer": {"taxable_income": 70000},
            }
        ]
        mock_cache_class.return_value = mock_cache
        
        explorer = CacheExplorer()
        output_path = Path("/tmp")
        
        with patch("builtins.open", create=True):
            result = explorer.export_cache_data(output_path)
        
        assert result is True
        mock_to_csv.assert_called_once()


class TestCacheComparison:
    """Test cache comparison functionality."""
    
    def test_compare_cached_vs_fresh(self):
        """Test comparing cached vs fresh responses."""
        # Mock client and results
        mock_client = MagicMock(spec=EDSLClient)
        
        # Mock survey
        mock_survey = MagicMock()
        mock_survey.questions = [MagicMock(text="Original question")]
        
        # Mock results
        mock_cached_result = MagicMock()
        mock_fresh_result = MagicMock()
        
        # Mock pandas DataFrames
        mock_cached_df = MagicMock()
        mock_cached_df.__getitem__.return_value.mean.return_value = 70000
        mock_cached_df.__getitem__.return_value.std.return_value = 5000
        
        mock_fresh_df = MagicMock()
        mock_fresh_df.__getitem__.return_value.mean.return_value = 71000
        mock_fresh_df.__getitem__.return_value.std.return_value = 4500
        
        mock_cached_result.select.return_value.to_pandas.return_value = mock_cached_df
        mock_fresh_result.select.return_value.to_pandas.return_value = mock_fresh_df
        
        # Set up client to return different results
        mock_client.run_survey.side_effect = [mock_cached_result, mock_fresh_result]
        
        # Run comparison
        comparison = compare_cached_vs_fresh(mock_client, mock_survey, n=10)
        
        assert comparison["cached_mean"] == 70000
        assert comparison["fresh_mean"] == 71000
        assert comparison["cached_std"] == 5000
        assert comparison["fresh_std"] == 4500
        assert comparison["n_samples"] == 10
        
        # Verify the survey was run twice with different questions
        assert mock_client.run_survey.call_count == 2