#!/usr/bin/env python3
"""
Demo simples do PyEhsa - Análise de Hotspots em São Paulo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from datetime import datetime, timedelta
import folium

# Importar PyEhsa do código fonte
from pyehsa.emerging_hotspot_analysis import EmergingHotspotAnalysis

# Configurar seed
np.random.seed(42)

# 1. Criar dados sintéticos - Grid 5x5 no centro de SP
center_lat, center_lon = -23.5489, -46.6388
step = 0.01  # ~1.1km

data = []
for i in range(5):
    for j in range(5):
        lat = center_lat + (i - 2) * step
        lon = center_lon + (j - 2) * step
        location_id = f'SP_{i}_{j}'
        
        # 6 meses de dados
        for month in range(6):
            time_period = datetime(2024, 1, 1) + timedelta(days=30*month)
            
            # Valor base
            value = np.random.poisson(10)
            
            # Simular hotspot emergente no canto superior direito
            if i >= 3 and j >= 3 and month >= 2:
                value += (month - 1) * 8
            
            # Ruído
            value += np.random.normal(0, 2)
            value = max(0, value)
            
            data.append({
                'location_id': location_id,
                'time_period': time_period,
                'value': value,
                'geometry': Point(lon, lat)
            })

# Criar GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:4326')

print(f"Dataset: {len(gdf)} observações, {gdf['location_id'].nunique()} locais")

# 2. Executar análise EHSA
results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
    gdf,
    region_id_field='location_id',
    time_period_field='time_period', 
    value='value',
    k=1,
    nsim=99
)

# 3. Mostrar resultados
print(f"\nResultados:")
print(results['classification'].value_counts())

# 4. Criar mapa simples
locations = gdf[['location_id', 'geometry']].drop_duplicates()
viz_data = results.merge(locations, left_on='region_id', right_on='location_id')

m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

color_map = {
    'no pattern detected': 'gray',
    'new hotspot': 'red',
    'consecutive hotspot': 'darkred',
    'sporadic hotspot': 'orange'
}

for _, row in viz_data.iterrows():
    color = color_map.get(row['classification'], 'gray')
    
    folium.CircleMarker(
        location=[row['geometry'].y, row['geometry'].x],
        radius=8,
        popup=f"{row['region_id']}: {row['classification']}",
        color='black',
        fillColor=color,
        fillOpacity=0.7
    ).add_to(m)

# Salvar mapa
m.save('hotspots_sao_paulo.html')
print(f"\nMapa salvo em: hotspots_sao_paulo.html")
