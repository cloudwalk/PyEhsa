import geopandas as gpd
import pandas as pd
import geohash
import itertools
from shapely import wkt
from shapely.ops import unary_union
from shapely.geometry import Polygon

# from pyehsa.pre_processing import PreProcessing
# PreProcessing.process()


def geohash_to_polygon(geohash_code):
    """
    Convert a geohash to a polygon using python-geohash and shapely.
    This replaces the polygon_geohasher.polygon_geohasher.geohash_to_polygon function.
    """
    # Decode geohash to get lat/lon and errors
    lat, lon, lat_err, lon_err = geohash.decode_exactly(geohash_code)

    # Create bounding box coordinates
    min_lat, max_lat = lat - lat_err, lat + lat_err
    min_lon, max_lon = lon - lon_err, lon + lon_err

    # Create polygon from bounding box
    return Polygon(
        [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]
    )


def geohashes_to_polygon(geohashes):
    """
    Local implementation that uses unary_union instead of the deprecated cascaded_union.
    This replaces the polygon_geohasher.polygon_geohasher.geohashes_to_polygon function
    to avoid deprecation warnings.
    """
    return unary_union([geohash_to_polygon(g) for g in geohashes])


class PreProcessing:

    @staticmethod
    def validate_and_clean_data(df, value_column, region_id_column=None):
        """
        Validate and clean input data for spatial analysis.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        value_column : str
            Name of the value column to analyze
        region_id_column : str, optional
            Name of the region identifier column

        Returns:
        --------
        pandas.DataFrame
            Cleaned dataframe with consistent data types
        """
        df_clean = df.copy()

        # Ensure value column is numeric float64
        if not pd.api.types.is_numeric_dtype(df_clean[value_column]):
            print(f"Converting {value_column} to numeric (float64)")
            df_clean[value_column] = pd.to_numeric(
                df_clean[value_column], errors="coerce"
            )

        df_clean[value_column] = df_clean[value_column].astype("float64")

        # Handle NaN values
        nan_count = df_clean[value_column].isna().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in {value_column}, filling with 0.0")
            df_clean[value_column] = df_clean[value_column].fillna(0.0)

        # Ensure region IDs are strings for consistency
        if region_id_column and region_id_column in df_clean.columns:
            df_clean[region_id_column] = df_clean[region_id_column].astype(str)

        return df_clean

    @staticmethod
    def complete_spacetime_cube(gdf, region_id_field, time_period_field):
        """
        Create a complete spacetime cube ensuring every location has data for every time period.
        This matches R sfdep's complete_spacetime_cube() function.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input GeoDataFrame with spatial and temporal data
        region_id_field : str
            Column name for region identifier
        time_period_field : str  
            Column name for time period
            
        Returns:
        --------
        GeoDataFrame
            Complete spacetime cube with NAs for missing combinations
        """
        # Get all unique locations and times
        all_locations = gdf[region_id_field].unique()
        all_times = gdf[time_period_field].unique()
        
        print(f"Creating complete spacetime cube:")
        print(f"  - {len(all_locations)} locations")
        print(f"  - {len(all_times)} time periods")
        print(f"  - {len(all_locations) * len(all_times)} total combinations")
        
        # Create all possible combinations
        all_combinations = list(itertools.product(all_locations, all_times))
        complete_df = pd.DataFrame(all_combinations, columns=[region_id_field, time_period_field])
        
        # Get geometry for each location (from the first occurrence)
        location_geometry = gdf.groupby(region_id_field).first()['geometry'].reset_index()
        
        # Merge geometry
        complete_df = complete_df.merge(location_geometry, on=region_id_field, how='left')
        
        # Merge with original data (this will create NAs for missing combinations)
        complete_df = complete_df.merge(
            gdf.drop(columns=['geometry']), 
            on=[region_id_field, time_period_field], 
            how='left'
        )
        
        # Convert back to GeoDataFrame
        complete_gdf = gpd.GeoDataFrame(complete_df, geometry='geometry')
        
        # Ensure CRS is preserved
        if gdf.crs is not None:
            complete_gdf = complete_gdf.set_crs(gdf.crs)
        
        print(f"  - Original data: {len(gdf)} rows")
        print(f"  - Complete cube: {len(complete_gdf)} rows")
        print(f"  - Missing combinations filled with NAs: {len(complete_gdf) - len(gdf)}")
        
        return complete_gdf

    @staticmethod
    def process(df):
        if not isinstance(df["geometry"].dtype, gpd.array.GeometryDtype):
            print("Applying WKT to geometry...")
            df["geometry"] = df["geometry"].apply(wkt.loads)

        sample_geom = df["geometry"].iloc[0]
        if not isinstance(sample_geom, str):
            print("Geometries are already Shapely objects, creating GeoDataFrame...")
            gdf = gpd.GeoDataFrame(df, geometry="geometry")
        else:
            print("Converting WKT strings to geometries...")
            df["geometry"] = df["geometry"].apply(wkt.loads)
            print("Creating GeoDataFrame...")
            gdf = gpd.GeoDataFrame(df, geometry="geometry")

        print("Setting CRS...")
        gdf.set_crs(epsg=4326, inplace=True)

        return gdf

    @staticmethod
    def get_geohash_geometries(df, geohash_field):
        polygons = []
        for geohash in df[f"{geohash_field}"]:
            if pd.notnull(geohash):
                polygon = geohashes_to_polygon([geohash])
                polygons.append(polygon)
            else:
                polygons.append(None)

        df["geometry"] = polygons

        df = gpd.GeoDataFrame(df, geometry="geometry")
        df.set_crs(epsg=4326, inplace=True)

        return df
