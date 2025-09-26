#!/usr/bin/env python3
"""
Basic Usage Example for PyEhsa

This script demonstrates how to use PyEhsa for Emerging Hot Spot Analysis
with synthetic spatio-temporal data.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import PyEhsa
from pyehsa import EmergingHotspotAnalysis, geohash_to_polygon

def create_synthetic_data(n_regions=16, n_time_periods=12, seed=42):
    """
    Create synthetic spatio-temporal data for demonstration.
    
    Parameters:
    -----------
    n_regions : int
        Number of spatial regions
    n_time_periods : int  
        Number of time periods
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Synthetic data with emerging hotspot patterns
    """
    np.random.seed(seed)
    
    # Create a 4x4 grid of regions
    grid_size = int(np.sqrt(n_regions))
    regions = []
    
    # Generate region data
    for i in range(grid_size):
        for j in range(grid_size):
            region_id = f"region_{i}_{j}"
            # Create point geometry
            x = j + np.random.normal(0, 0.1)  # Add small noise
            y = i + np.random.normal(0, 0.1)
            regions.append({
                'region_id': region_id,
                'geometry': Point(x, y),
                'grid_x': j,
                'grid_y': i
            })
    
    # Generate time series data
    base_date = datetime(2023, 1, 1)
    data = []
    
    for region in regions:
        # Determine if this region will be a hotspot
        is_hotspot_region = (region['grid_x'] + region['grid_y']) % 3 == 0
        
        for t in range(n_time_periods):
            # Create time period
            time_period = base_date + timedelta(days=30 * t)  # Monthly data
            
            # Generate base value
            base_value = np.random.poisson(5)  # Base count
            
            if is_hotspot_region:
                # Create emerging hotspot pattern (increases over time)
                trend = max(0, t - n_time_periods // 2) * 0.5
                hotspot_boost = np.random.poisson(trend) if t > n_time_periods // 3 else 0
                value = base_value + hotspot_boost
            else:
                # Normal variation
                value = base_value + np.random.normal(0, 1)
                
            # Ensure non-negative values
            value = max(0, value)
            
            data.append({
                'region_id': region['region_id'],
                'time_period': time_period,
                'value': value,
                'geometry': region['geometry'],
                'is_synthetic_hotspot': is_hotspot_region
            })
    
    return gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')


def run_basic_example():
    """Run basic PyEhsa example."""
    print("🚀 PyEhsa Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Create synthetic data
    print("📊 Step 1: Creating synthetic spatio-temporal data...")
    data = create_synthetic_data(n_regions=16, n_time_periods=8, seed=42)
    
    print(f"   - Created dataset with {len(data)} observations")
    print(f"   - {data['region_id'].nunique()} regions")
    print(f"   - {data['time_period'].nunique()} time periods")
    print(f"   - Value range: [{data['value'].min():.1f}, {data['value'].max():.1f}]")
    print()
    
    # Display sample data
    print("📋 Sample of synthetic data:")
    print(data[['region_id', 'time_period', 'value', 'is_synthetic_hotspot']].head(10))
    print()
    
    # Step 2: Run EHSA analysis
    print("🔥 Step 2: Running Emerging Hot Spot Analysis...")
    try:
        results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
            data,
            region_id_field='region_id',
            time_period_field='time_period',
            value='value',
            k=1,           # 1 time lag
            nsim=99        # 99 simulations (reduced for demo)
        )
        
        print("✅ EHSA analysis completed successfully!")
        print()
        
        # Step 3: Display results
        print("📋 EHSA Results:")
        print("-" * 40)
        print(f"Total regions analyzed: {len(results)}")
        
        # Classification distribution
        if 'classification' in results.columns:
            print("\n🏷️  Classification distribution:")
            class_counts = results['classification'].value_counts()
            for classification, count in class_counts.items():
                percentage = (count / len(results)) * 100
                print(f"   - {classification}: {count} regions ({percentage:.1f}%)")
        
        # Show detailed results for hotspots
        if 'classification' in results.columns:
            hotspots = results[results['classification'] != 'no pattern detected']
            if len(hotspots) > 0:
                print(f"\n🔥 Detected hotspots ({len(hotspots)} regions):")
                for _, row in hotspots.iterrows():
                    print(f"   - {row['region_id']}: {row['classification']}")
            else:
                print("\n⭕ No significant hotspot patterns detected")
        
        # Statistical summary
        if 'tau' in results.columns:
            print(f"\n📊 Mann-Kendall Tau statistics:")
            print(f"   - Mean: {results['tau'].mean():.3f}")
            print(f"   - Std: {results['tau'].std():.3f}")
            print(f"   - Range: [{results['tau'].min():.3f}, {results['tau'].max():.3f}]")
        
        print("\n" + "=" * 50)
        print("✅ Example completed successfully!")
        print("\n💡 This demonstrates PyEhsa's ability to detect emerging")
        print("   hotspot patterns in spatio-temporal data.")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during EHSA analysis: {e}")
        print("   This might happen with synthetic data that has")
        print("   insufficient spatial relationships.")
        return None


def demonstrate_preprocessing():
    """Demonstrate preprocessing functions."""
    print("\n" + "=" * 50)  
    print("🔧 PyEhsa Preprocessing Functions Demo")
    print("=" * 50)
    
    # Geohash conversion example
    print("📍 Geohash to Polygon Conversion:")
    sample_geohash = "9q8yy"  # San Francisco area
    polygon = geohash_to_polygon(sample_geohash)
    
    print(f"   - Geohash: {sample_geohash}")
    print(f"   - Polygon area: {polygon.area:.6f}")
    print(f"   - Polygon bounds: {polygon.bounds}")
    print()
    
    # Data validation example
    from pyehsa.pre_processing import PreProcessing
    
    print("🧹 Data Validation Example:")
    dirty_data = pd.DataFrame({
        'region': ['A', 'B', 'C'],
        'values': [1.5, np.nan, '2.5'],  # Mixed types, NaN
        'time': ['2023-01', '2023-02', '2023-03']
    })
    
    print("   Before cleaning:")
    print(f"   - Data types: {dirty_data.dtypes.to_dict()}")
    print(f"   - NaN count: {dirty_data.isna().sum().sum()}")
    
    clean_data = PreProcessing.validate_and_clean_data(
        dirty_data, 'values', 'region'
    )
    
    print("   After cleaning:")
    print(f"   - Data types: {clean_data.dtypes.to_dict()}")
    print(f"   - NaN count: {clean_data.isna().sum().sum()}")
    print("   ✅ Data cleaned and validated!")


if __name__ == "__main__":
    # Run the main example
    results = run_basic_example()
    
    # Run preprocessing demo
    demonstrate_preprocessing()
    
    print("\n" + "🎉" * 20)
    print("PyEhsa example completed! ")
    print("Check the documentation for more advanced usage.")
    print("🎉" * 20)
