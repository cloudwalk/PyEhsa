import geopandas as gpd
import numpy as np
from libpysal.weights import Queen


class SpacialWeights:
    @staticmethod
    def calculate_spatial_weights(gdf, region_id_field):
        """
        Calculates spatial weights for the entire dataset based on unique geometries.
        Matches R sfdep implementation:
        1. Uses Queen contiguity 
        2. Includes self as neighbor (like R's include_self())
        3. Row standardized weights
        """
        unique_geoms = gdf.drop_duplicates(subset=region_id_field)
        
        # Ensure the index of unique_geoms matches the region_id for proper alignment
        unique_geoms = unique_geoms.set_index(region_id_field)

        # Calculate Queen contiguity weights. This defines neighbors as polygons
        # that share at least one vertex (corners or edges).
        w = Queen.from_dataframe(unique_geoms, use_index=True)
        
        # Include self as neighbor (matching R's include_self() function)
        # This is critical for matching R results
        for i in w.id_order:
            if i not in w.neighbors[i]:
                # Add self to neighbors list
                w.neighbors[i] = w.neighbors[i] + [i]
                # Add weight of 1.0 for self  
                w.weights[i] = w.weights[i] + [1.0]
        
        # Apply row standardization (matching R's row standardized weights)
        w.transform = 'r'
        
        return w
        
    @staticmethod
    def create_spacetime_neighbors_and_weights(w, n_times, n_locs, k=1):
        """
        Create time-lagged spacetime neighbors and weights.
        Matches R's spt_nb() and spt_wt() functions.
        
        Parameters:
        -----------
        w : PySAL weights object
            Base spatial weights from unique geometries
        n_times : int
            Number of time periods
        n_locs : int  
            Number of unique locations
        k : int
            Number of time lags to include (default 1, matching R default)
            
        Returns:
        --------
        tuple : (spacetime_neighbors, spacetime_weights)
            Time-lagged neighbors and weights for the full spacetime cube
        """
        # Create spacetime neighbors list 
        spt_neighbors = {}
        spt_weights = {}
        
        # Performance optimization: Create lookup dictionary to avoid repeated list.index() calls
        # This changes O(n) lookup to O(1), preventing algorithmic DoS on large datasets
        id_to_idx = {loc_id: idx for idx, loc_id in enumerate(w.id_order)}
        
        # For each time period and location, create neighbors including time lags
        for t in range(n_times):
            for i, loc_id in enumerate(w.id_order):
                # Calculate spacetime index: location i at time t
                st_idx = t * n_locs + i
                
                neighbors_list = []
                weights_list = []
                
                # Add spatial neighbors from current time period
                for j, neighbor_id in enumerate(w.neighbors[loc_id]):
                    neighbor_loc_idx = id_to_idx[neighbor_id]  # O(1) lookup instead of O(n)
                    neighbor_st_idx = t * n_locs + neighbor_loc_idx
                    neighbors_list.append(neighbor_st_idx)
                    weights_list.append(w.weights[loc_id][j])
                
                # Add time-lagged neighbors (from k previous time periods)
                for lag in range(1, k + 1):
                    if t >= lag:  # Only if we have enough previous time periods
                        lag_t = t - lag
                        for j, neighbor_id in enumerate(w.neighbors[loc_id]):
                            neighbor_loc_idx = id_to_idx[neighbor_id]  # O(1) lookup instead of O(n)
                            neighbor_st_idx = lag_t * n_locs + neighbor_loc_idx
                            neighbors_list.append(neighbor_st_idx)
                            weights_list.append(w.weights[loc_id][j])
                
                spt_neighbors[st_idx] = neighbors_list
                spt_weights[st_idx] = weights_list
        
        return spt_neighbors, spt_weights
