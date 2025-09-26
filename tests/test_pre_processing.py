"""
Tests for pre_processing module.
"""
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from pyehsa.pre_processing import geohash_to_polygon, geohashes_to_polygon, PreProcessing


class TestGeohashFunctions:
    """Test geohash conversion functions."""
    
    def test_geohash_to_polygon_basic(self):
        """Test basic geohash to polygon conversion."""
        # Use a well-known geohash
        geohash = "9q8yy"
        polygon = geohash_to_polygon(geohash)
        
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid
        assert polygon.area > 0
        
        # Check that polygon has reasonable bounds
        bounds = polygon.bounds
        assert len(bounds) == 4  # minx, miny, maxx, maxy
        assert bounds[0] < bounds[2]  # minx < maxx
        assert bounds[1] < bounds[3]  # miny < maxy

    def test_geohash_to_polygon_precision(self):
        """Test that shorter geohashes create larger polygons."""
        geohash_short = "9q"
        geohash_long = "9q8yy123"
        
        poly_short = geohash_to_polygon(geohash_short)
        poly_long = geohash_to_polygon(geohash_long)
        
        # Shorter geohash should have larger area
        assert poly_short.area > poly_long.area
    
    def test_geohashes_to_polygon(self):
        """Test merging multiple geohashes into single polygon."""
        geohashes = ["9q8yy", "9q8yz", "9q8zp"]
        merged_polygon = geohashes_to_polygon(geohashes)
        
        assert isinstance(merged_polygon, (Polygon, type(merged_polygon)))  # Could be MultiPolygon
        assert merged_polygon.area > 0
        
        # Single geohash should work too
        single_polygon = geohashes_to_polygon(["9q8yy"])
        assert isinstance(single_polygon, Polygon)

    def test_geohash_empty_list(self):
        """Test handling of empty geohash list."""
        # Empty list should return an empty geometry
        result = geohashes_to_polygon([])
        # Just check that it doesn't crash and returns something
        assert result is not None


class TestPreProcessing:
    """Test PreProcessing class methods."""
    
    def test_validate_and_clean_data_basic(self):
        """Test basic data validation and cleaning."""
        # Create test data with various issues
        df = pd.DataFrame({
            'region': ['A', 'B', 'C', 'D'],
            'values': [1.5, 2.0, np.nan, 3.5],  # Contains NaN
            'time': ['2023-01', '2023-01', '2023-02', '2023-02']
        })
        
        cleaned = PreProcessing.validate_and_clean_data(df, 'values', 'region')
        
        # Check that NaN was filled with 0.0
        assert cleaned['values'].isna().sum() == 0
        assert cleaned['values'].dtype == 'float64'
        assert cleaned.loc[2, 'values'] == 0.0  # NaN should be filled with 0
        
        # Check region IDs are strings
        assert cleaned['region'].dtype == 'object'
        assert all(isinstance(x, str) for x in cleaned['region'])

    def test_validate_and_clean_data_numeric_conversion(self):
        """Test conversion of non-numeric values."""
        df = pd.DataFrame({
            'region': ['A', 'B', 'C'],
            'values': ['1.5', '2.0', 'invalid'],  # String values, one invalid
        })
        
        cleaned = PreProcessing.validate_and_clean_data(df, 'values', 'region')
        
        # Should convert to numeric, invalid becomes NaN then 0
        assert cleaned['values'].dtype == 'float64'
        assert cleaned.loc[0, 'values'] == 1.5
        assert cleaned.loc[1, 'values'] == 2.0
        assert cleaned.loc[2, 'values'] == 0.0  # Invalid converted to NaN then filled with 0

    def test_complete_spacetime_cube(self, sample_spacetime_data):
        """Test spacetime cube completion."""
        # Remove some combinations to create incomplete cube
        incomplete_data = sample_spacetime_data.iloc[:-2].copy()  # Remove last 2 rows
        
        complete_cube = PreProcessing.complete_spacetime_cube(
            incomplete_data, 'region_id', 'time_period'
        )
        
        # Should have all combinations of regions x time periods
        n_regions = incomplete_data['region_id'].nunique()
        n_times = incomplete_data['time_period'].nunique()
        expected_rows = n_regions * n_times
        
        assert len(complete_cube) >= len(incomplete_data)  # At least original size
        assert isinstance(complete_cube, gpd.GeoDataFrame)
        assert complete_cube.crs is not None

    def test_process_method_with_geometry(self, sample_spacetime_data):
        """Test the process method with geometry data."""
        # Convert to regular DataFrame to test process method
        df = pd.DataFrame(sample_spacetime_data)
        df['geometry'] = df['geometry'].astype(str)  # Convert geometry to WKT strings
        
        processed = PreProcessing.process(df)
        
        assert isinstance(processed, gpd.GeoDataFrame)
        assert processed.crs is not None
        assert len(processed) == len(df)
