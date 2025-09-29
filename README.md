# PyEhsa: Emerging Hot Spot Analysis in Python

[![PyPI version](https://badge.fury.io/py/pyehsa.svg)](https://badge.fury.io/py/pyehsa)
[![Python](https://img.shields.io/pypi/pyversions/pyehsa.svg)](https://pypi.org/project/pyehsa)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PyEhsa** is a Python library for **Emerging Hot Spot Analysis (EHSA)** of spatio-temporal data, providing functionality similar to R's `sfdep` package. It enables researchers and analysts to identify and classify spatial-temporal patterns, emerging hotspots, and coldspots in geographic data.

## 🚀 Key Features

- **Emerging Hot Spot Analysis**: Identify emerging patterns in spatio-temporal data
- **Spatial Statistics**: Calculate Getis-Ord Gi* statistics for hotspot detection  
- **Mann-Kendall Trend Analysis**: Detect trends in time series data
- **Multiple Classifications**: Comprehensive hotspot pattern classification
- **Visualization Tools**: Create interactive maps and plots with Folium and Plotly
- **Flexible Data Input**: Support for pandas DataFrames and GeoPandas GeoDataFrames

## 📦 Installation

```bash
pip install pyehsa
```

## 🔧 Requirements

- Python 3.9+
- pandas >= 2.0.0
- geopandas >= 0.14.0  
- numpy >= 1.24.0
- scipy >= 1.10.0
- libpysal >= 4.9.0

## 📖 Quick Start

```python
import pandas as pd
import geopandas as gpd
from pyehsa import EmergingHotspotAnalysis

# Load your spatio-temporal data
df = pd.read_csv("your_data.csv")  
geo = gpd.read_file("your_shapes.geojson")

# Merge data with geometry
data = df.merge(geo[['region_id', 'geometry']], on='region_id')
data = gpd.GeoDataFrame(data, geometry='geometry')
data = data.set_crs(epsg=4326)

# Convert time column to datetime
data['time_period'] = pd.to_datetime(data['time_period'])

# Run Emerging Hot Spot Analysis
results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
    data, 
    location_col='region_id', 
    time_col='time_period', 
    value_col='value',
    k=1,      # spatial neighbors
    nsim=199  # Monte Carlo simulations
)

# View results
print(results['classification'].value_counts())
```

## 📊 Example Output

The analysis returns a DataFrame with classifications such as:
- **New Hot Spot**: Recently emerged statistically significant hotspots
- **Consecutive Hot Spot**: Hotspots that are statistically significant for multiple consecutive time periods  
- **Oscillating Hot Spot**: Areas that are statistically significant hotspots for some periods but not others
- **Cold Spot**: Areas with statistically significant low values
- **No Pattern Detected**: Areas without statistically significant patterns

## 🗺️ Visualization

```python
from pyehsa import EhsaPlotting

# Create interactive map
plotter = EhsaPlotting()
map_viz = plotter.plot_ehsa_map(results, 'classification')

# Save or display
map_viz.save('hotspot_analysis.html')
```

## 📚 Documentation

For detailed documentation, examples, and API reference, visit: [https://pyehsa.readthedocs.io](https://pyehsa.readthedocs.io)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 Citation

If you use PyEhsa in your research, please cite:

```bibtex
@software{pyehsa2025,
  author = {CloudWalk},
  title = {PyEhsa: Emerging Hot Spot Analysis in Python},
  year = {2025},
  url = {https://github.com/cloudwalk/PyEhsa}
}
```

## 📋 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Related Work

PyEhsa is inspired by and provides similar functionality to:
- R's `sfdep` package for spatial dependence analysis
- ArcGIS's Emerging Hot Spot Analysis tool
- PySAL ecosystem for spatial analysis

## 📈 Development Status

This package is actively developed and maintained. For support, please open an issue on [GitHub Issues](https://github.com/cloudwalk/PyEhsa/issues).

---
