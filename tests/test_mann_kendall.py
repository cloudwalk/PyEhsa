"""
Tests for Mann-Kendall trend analysis.
"""
import pytest
import numpy as np

from pyehsa.mann_kendall import MannKendall


class TestMannKendall:
    """Test Mann-Kendall trend analysis."""
    
    def test_mann_kendall_increasing_trend(self, sample_mann_kendall_data):
        """Test Mann-Kendall with clear increasing trend."""
        data = sample_mann_kendall_data['increasing_trend']
        result = MannKendall.mann_kendall_test(data)
        
        assert isinstance(result, dict)
        assert 'tau' in result
        assert 'sl' in result  # significance level (p-value)
        assert 'S' in result
        assert 'D' in result
        assert 'varS' in result
        
        # For increasing trend, tau should be positive
        assert result['tau'] > 0
        # p-value should be small for clear trend
        assert result['sl'] < 0.05
        
    def test_mann_kendall_decreasing_trend(self, sample_mann_kendall_data):
        """Test Mann-Kendall with clear decreasing trend."""
        data = sample_mann_kendall_data['decreasing_trend']
        result = MannKendall.mann_kendall_test(data)
        
        # For decreasing trend, tau should be negative
        assert result['tau'] < 0
        # p-value should be small for clear trend
        assert result['sl'] < 0.05
    
    def test_mann_kendall_no_trend(self, sample_mann_kendall_data):
        """Test Mann-Kendall with no trend (constant values)."""
        data = sample_mann_kendall_data['no_trend']
        result = MannKendall.mann_kendall_test(data)
        
        # For no trend, tau should be close to 0
        assert abs(result['tau']) < 0.1
        # p-value should be large (not significant)
        assert result['sl'] > 0.05
    
    def test_mann_kendall_mixed_data(self, sample_mann_kendall_data):
        """Test Mann-Kendall with mixed trend data."""
        data = sample_mann_kendall_data['mixed_data']
        result = MannKendall.mann_kendall_test(data)
        
        # Should return valid result
        assert isinstance(result['tau'], float)
        assert isinstance(result['sl'], float)
        assert 0 <= result['sl'] <= 1  # p-value should be between 0 and 1
    
    def test_mann_kendall_insufficient_data(self):
        """Test Mann-Kendall with insufficient data points."""
        # Single data point
        result = MannKendall.mann_kendall_test([5.0])
        assert result['tau'] == 0
        assert result['sl'] == 1.0
        
        # Empty data
        result = MannKendall.mann_kendall_test([])
        assert result['tau'] == 0
        assert result['sl'] == 1.0
    
    def test_mann_kendall_with_nan_values(self):
        """Test Mann-Kendall with NaN values in data."""
        data = [1.0, 2.0, np.nan, 4.0, 5.0, np.nan]
        result = MannKendall.mann_kendall_test(data)
        
        # Should handle NaN values gracefully
        assert isinstance(result['tau'], float)
        assert isinstance(result['sl'], float)
        assert not np.isnan(result['tau'])
        assert not np.isnan(result['sl'])
    
    def test_mann_kendall_with_ties(self):
        """Test Mann-Kendall with tied values."""
        data = [1, 2, 2, 3, 3, 3, 4]  # Contains ties
        result = MannKendall.mann_kendall_test(data)
        
        # Should handle tied values and adjust variance
        assert isinstance(result['tau'], float)
        assert isinstance(result['sl'], float)
        assert result['varS'] >= 0  # Variance should be non-negative
    
    def test_mann_kendall_return_format(self):
        """Test that Mann-Kendall returns properly formatted results."""
        data = [1, 3, 2, 4, 5]
        result = MannKendall.mann_kendall_test(data)
        
        # Check all required keys are present
        required_keys = ['x', 'tau', 'sl', 'S', 'D', 'varS']
        for key in required_keys:
            assert key in result
        
        # Check data types
        assert isinstance(result['tau'], float)
        assert isinstance(result['sl'], float)
        assert isinstance(result['S'], int)
        assert isinstance(result['D'], (int, float))
        assert isinstance(result['varS'], float)
