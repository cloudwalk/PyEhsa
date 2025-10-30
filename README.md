# PyEhsa: Emerging Hot Spot Analysis in Python

[![PyPI version](https://badge.fury.io/py/pyehsa.svg)](https://badge.fury.io/py/pyehsa)
[![Python](https://img.shields.io/pypi/pyversions/pyehsa.svg)](https://pypi.org/project/pyehsa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/cloudwalk/PyEhsa/workflows/Tests/badge.svg)](https://github.com/cloudwalk/PyEhsa/actions)

**PyEhsa** is a Python library for **Emerging Hot Spot Analysis (EHSA)** of spatio-temporal data, providing functionality similar to R's `sfdep` package. It enables researchers and analysts to identify and classify spatial-temporal patterns, emerging hotspots, and coldspots in geographic data.

## 🚀 Key Features

- **Emerging Hot Spot Analysis**: Identify emerging patterns in spatio-temporal data
- **Spatial Statistics**: Calculate Getis-Ord Gi* statistics for hotspot detection  
- **Mann-Kendall Trend Analysis**: Detect trends in time series data
- **Multiple Classifications**: Comprehensive hotspot pattern classification
- **Interactive Visualization**: HTML tool with time series analysis and mapping
- **Flexible Data Input**: Support for pandas DataFrames and GeoPandas GeoDataFrames

## 📦 Installation

```bash
pip install pyehsa
```

## 🔧 Requirements

- Python 3.9+
- pandas >= 1.5.0
- geopandas >= 0.13.0  
- numpy >= 1.21.0
- scipy >= 1.9.0
- libpysal >= 4.6.0

## 📖 Quick Start Guide

### Step 1: Import Libraries

```python
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from pyehsa import EmergingHotspotAnalysis
```

### Step 2: Prepare Your Data

Your dataset should have:
- **`region_id`**: Spatial identifier (e.g., 'location_001', 'geohash_abc')
- **`time_period`**: Temporal column (datetime format)
- **`value`**: Numeric variable to analyze
- **`geometry`**: Spatial geometry (Point or Polygon from shapely)

```python
# Example: Create or load your GeoDataFrame
data = pd.DataFrame({
    'region_id': ['area_1', 'area_1', 'area_2', 'area_2'],
    'time_period': [datetime(2024, 1, 1), datetime(2024, 2, 1), 
                    datetime(2024, 1, 1), datetime(2024, 2, 1)],
    'value': [10.5, 15.2, 8.3, 12.1],
    'geometry': [polygon_1, polygon_1, polygon_2, polygon_2]  # Your geometries
})

gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')
```

### Step 3: Run Emerging Hot Spot Analysis

```python
results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
    gdf,
    region_id_field='region_id',
    time_period_field='time_period',
    value='value',
    k=1,      # Temporal lags to include
    nsim=99   # Monte Carlo simulations for significance testing
)
```

### Step 4: View Results

```python
# See classification distribution
print(results['classification'].value_counts())

# View detailed results
results.head()
```

### Step 5: Create Interactive Visualization

```python
from pyehsa import EhsaPlotting

# Merge geometry with results for visualization
locations = gdf[['region_id', 'geometry']].drop_duplicates()
viz_data = results.merge(locations, left_on=results.columns[0], right_on='region_id')

# Create interactive map
ehsa_map = EhsaPlotting.plot_ehsa_map_interactive(
    df=viz_data,
    region_id_field='region_id',
    title="Emerging Hotspots Analysis"
)
ehsa_map.show()
```

## 📊 Classification Types

The analysis returns classifications such as:
- **new hotspot**: Recently emerged statistically significant hotspots
- **consecutive hotspot**: Hotspots significant for multiple consecutive periods  
- **sporadic hotspot**: Intermittent hotspot patterns
- **intensifying hotspot**: Hotspots with increasing trend
- **persistent hotspot**: Long-term stable hotspots
- **oscillating hotspot**: Alternating hotspot patterns
- **historical hotspot**: Previously significant hotspots
- **coldspot variations**: Similar patterns for low-value clusters
- **no pattern detected**: Areas without significant patterns

## 🛠️ Advanced Usage

### Interactive HTML Visualization Tool

```python
# Launch interactive visualization tool
EmergingHotspotAnalysis.launch_visualization(results)

# Or save to file
EmergingHotspotAnalysis.save_visualization('my_analysis.html', results)
```

The HTML tool provides:
- Interactive map with polygon visualization
- Time series charts for each region
- Classification details and statistics
- Mann-Kendall trend analysis results

### Static Map Visualization

```python
plotter = EhsaPlotting()
fig = plotter.plot_ehsa_map_static(viz_data, title="My Analysis")
fig.savefig('hotspots_map.png', dpi=300, bbox_inches='tight')
```

## 📚 Complete Example

See `examples/demo_sp.ipynb` for a complete working example with:
- Synthetic grid data covering São Paulo
- Full EHSA workflow from data creation to visualization
- Interactive map generation

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Citation

If you use PyEhsa in your research, please cite:

```bibtex
@software{pyehsa2025,
  author = {CloudWalk},
  title = {PyEhsa: Emerging Hot Spot Analysis in Python},
  year = {2025},
  url = {https://github.com/cloudwalk/PyEhsa},
  version = {0.1.0}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Related Work

PyEhsa is inspired by and provides similar functionality to:
- R's `sfdep` package for spatial dependence analysis
- ArcGIS's Emerging Hot Spot Analysis tool
- PySAL ecosystem for spatial analysis

## 📈 Development & Support

- **Repository**: [github.com/cloudwalk/PyEhsa](https://github.com/cloudwalk/PyEhsa)
- **Issues**: [GitHub Issues](https://github.com/cloudwalk/PyEhsa/issues)
- **PyPI**: [pypi.org/project/pyehsa](https://pypi.org/project/pyehsa)

## 🙏 Acknowledgments

Developed by CloudWalk's data science team to provide open-source spatial analysis tools for the Python community.