"""
Pytest configuration and shared fixtures for PyEhsa tests.
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from datetime import datetime, timedelta


@pytest.fixture
def sample_geohash_data():
    """Create synthetic data with known geohashes for testing."""
    return {
        'region_id': ['9q8yy', '9q8yz', '9q8zp', '9q8zr'],
        'value': [1.5, 2.3, 0.8, 3.1],
        'time_period': ['2023-01', '2023-01', '2023-01', '2023-01']
    }


@pytest.fixture  
def sample_spacetime_data():
    """Create synthetic spatio-temporal data for EHSA testing."""
    np.random.seed(42)  # For reproducible results
    
    # Create a small grid of regions
    regions = ['reg_A', 'reg_B', 'reg_C', 'reg_D']
    dates = pd.date_range('2023-01-01', periods=6, freq='ME')
    
    data = []
    for region in regions:
        for date in dates:
            # Create synthetic values with some patterns
            base_value = hash(region) % 10  # Base value per region
            time_trend = dates.get_loc(date) * 0.1  # Slight time trend
            noise = np.random.normal(0, 0.5)  # Random noise
            
            value = base_value + time_trend + noise
            
            data.append({
                'region_id': region,
                'time_period': date,
                'value': max(0, value),  # Ensure non-negative
                'geometry': Point(
                    hash(region) % 100 / 100,  # Synthetic coordinates
                    hash(region[::-1]) % 100 / 100
                )
            })
    
    return gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')


@pytest.fixture
def sample_mann_kendall_data():
    """Create data series for Mann-Kendall testing."""
    return {
        'increasing_trend': [1, 2, 3, 4, 5, 6],  # Clear upward trend
        'decreasing_trend': [6, 5, 4, 3, 2, 1],  # Clear downward trend  
        'no_trend': [3, 3, 3, 3, 3, 3],         # No trend
        'mixed_data': [1, 3, 2, 4, 3, 5]        # Mixed trend
    }


@pytest.fixture
def sample_polygon():
    """Create a sample polygon for geometric tests."""
    return Polygon([
        (-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)
    ])
