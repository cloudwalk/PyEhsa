"""
PyEhsa: A Python library for Emerging Hot Spot Analysis (EHSA) of spatio-temporal data

PyEhsa provides tools for analyzing spatio-temporal patterns and emerging hotspots
in geographic data, similar to functionality found in R's sfdep package.
"""

__version__ = "0.1.0"
__author__ = "CloudWalk"
__email__ = "lucas.azevedo@cloudwalk.io"

# Import main classes for easy access
from .emerging_hotspot_analysis import EmergingHotspotAnalysis
from .ehsa_classification import EhsaClassification
from .ehsa_plotting import EhsaPlotting
from .gi_star import GiStar
from .mann_kendall import MannKendall
from .pre_processing import (
    geohash_to_polygon,
    geohashes_to_polygon, 
    PreProcessing
)
from .spacial_weights import SpacialWeights

# Define what gets imported with "from pyehsa import *"
__all__ = [
    "EmergingHotspotAnalysis",
    "EhsaClassification", 
    "EhsaPlotting", 
    "GiStar",
    "MannKendall",
    "geohash_to_polygon",
    "geohashes_to_polygon",
    "PreProcessing",
    "SpacialWeights"
]
