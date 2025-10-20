import pandas as pd
import numpy as np
import warnings
from .spacial_weights import SpacialWeights


class GiStar:
    @staticmethod  
    def calculate_gi_star_spacetime(gdf, w, region_id_field, time_period_field, value, k=1, nsim=199):
        """
        Calculate Gi* statistics for spacetime data matching R sfdep implementation EXACTLY.
        This is a complete rewrite to match R's local_g_spt function.
        
        Parameters:
        -----------
        gdf : GeoDataFrame
            Input geodataframe with spatial and temporal data
        w : PySAL weights object
            Spatial weights (should include self as neighbor)
        region_id_field : str
            Column name for region identifier  
        time_period_field : str
            Column name for time period
        value : str
            Column name for values to analyze
        k : int
            Number of time lags (default 1, matching R)
        nsim : int
            Number of simulations for p-values (default 199, matching R default)
            
        Returns:
        --------
        GeoDataFrame
            Original data with gi_star and p_sim columns added
        """
        # Validate nsim parameter to prevent resource exhaustion
        if not isinstance(nsim, int) or nsim < 1:
            raise ValueError(f"nsim must be a positive integer, got: {nsim}")
        if nsim > 9999:
            raise ValueError(f"nsim is too large ({nsim}). Maximum allowed: 9999")
        if nsim > 999:
            warnings.warn(f"nsim={nsim} is large and may consume significant resources. "
                         f"Typical values are 99-999.")
        
        # Get sorted data (CRITICAL: must match R's ordering)
        gdf_sorted = gdf.sort_values([time_period_field, region_id_field]).reset_index(drop=True)
        
        # Get dimensions
        unique_regions = gdf_sorted[region_id_field].unique()
        unique_times = sorted(gdf_sorted[time_period_field].unique())
        n_locs = len(unique_regions) 
        n_times = len(unique_times)
        
        # Extract values as array (matching R's spacetime cube order: time-major)
        values = gdf_sorted[value].values.astype(float)
        
        print(f"Spacetime cube dimensions:")
        print(f"  - {n_locs} locations")
        print(f"  - {n_times} time periods") 
        print(f"  - {len(gdf_sorted)} total observations")
        
        # Create spacetime neighbors list (matching R's spt_nb function)
        spt_nb = GiStar._create_spacetime_neighbors_r_style(w, n_times, n_locs, k)
        spt_wt = GiStar._create_spacetime_weights_r_style(w, spt_nb, n_times, n_locs, k)
        
        # Debug: Check for potential out-of-bounds issues
        max_neighbor_idx = max(max(neighbors) if neighbors else 0 for neighbors in spt_nb)
        if max_neighbor_idx >= len(values):
            print(f"WARNING: Max neighbor index {max_neighbor_idx} exceeds data size {len(values)}")
            print(f"This will be handled by bounds checking in _find_xj_r_style")
        
        # Calculate observed Gi* using R's exact local_g_spt implementation
        observed_gi = GiStar._local_g_spt_r_exact(values, unique_times, spt_nb, spt_wt, n_locs)
        
        # Calculate p-values using R's exact simulation approach  
        p_values = GiStar._calculate_p_values_r_exact(values, unique_times, spt_nb, spt_wt, n_locs, observed_gi, nsim)
        
        # Summary stats for verification
        print(f"Gi* range: [{np.nanmin(observed_gi):.3f}, {np.nanmax(observed_gi):.3f}]")
        print(f"Significant observations (p<=0.01): {np.sum(p_values <= 0.01)}/{len(p_values)}")
        print(f"Significant observations (p<=0.05): {np.sum(p_values <= 0.05)}/{len(p_values)}")
        
        # DEBUG: Check some specific p-values vs Gi* values  
        print(f"DEBUG - Sample of borderline cases:")
        for i in range(0, min(50, len(p_values)), 10):
            print(f"  Obs {i}: Gi*={observed_gi[i]:.3f}, p={p_values[i]:.4f}, sig={p_values[i] <= 0.01}")
            
        # DEBUG: Neighbor permutation test
        print(f"DEBUG - Neighbor permutation test:")
        test_nb_orig = spt_nb[:3]  # First 3 neighbor lists
        test_nb_perm = GiStar._conditional_permute_neighbors_r_style(spt_nb[:10], n_locs)[:3]
        print(f"  Original neighbors: {test_nb_orig}")
        print(f"  Permuted neighbors: {test_nb_perm}")
        print(f"  Neighbor structure changed: {test_nb_orig != test_nb_perm}")
        
        # Add results to dataframe
        gdf_sorted['gi'] = observed_gi
        gdf_sorted['p_sim'] = p_values
        
        # Create cluster classifications (matching R's threshold = 0.01!)
        p_threshold = 0.01  # CRITICAL: R uses 0.01, not 0.05!
        clusters = np.full(len(observed_gi), 'Not Significant', dtype=object)
        hot_spots = (p_values <= p_threshold) & (observed_gi > 0)  # Use <= like R
        cold_spots = (p_values <= p_threshold) & (observed_gi < 0)  # Use <= like R  
        clusters[hot_spots] = 'Hot Spot'
        clusters[cold_spots] = 'Cold Spot'
        gdf_sorted['cluster'] = clusters
        
        return gdf_sorted
    
    @staticmethod
    def _create_spacetime_neighbors_r_style(w, n_times, n_locs, k):
        """
        Create spacetime neighbors list exactly matching R's spt_nb function.
        """
        # Get base spatial neighbors from weights object
        nb_base = []
        for i in range(n_locs):
            region_id = w.id_order[i]
            neighbors = w.neighbors[region_id]
            # Convert neighbor IDs to indices
            neighbor_indices = [w.id_order.index(nid) for nid in neighbors]
            nb_base.append(neighbor_indices)
        
        # Create spacetime neighbors list
        spt_nb = []
        
        # For each time period and location
        for t in range(n_times):
            for i in range(n_locs):
                neighbors = []
                
                # Add spatial neighbors from current time period
                for neighbor_idx in nb_base[i]:
                    st_neighbor_idx = t * n_locs + neighbor_idx
                    neighbors.append(st_neighbor_idx)
                
                # Add time-lagged neighbors (from k previous time periods)
                for lag in range(1, k + 1):
                    if t >= lag:  # Only if we have enough previous time periods
                        lag_t = t - lag
                        for neighbor_idx in nb_base[i]:
                            st_neighbor_idx = lag_t * n_locs + neighbor_idx
                            neighbors.append(st_neighbor_idx)
                
                spt_nb.append(neighbors)
        
        return spt_nb
    
    @staticmethod
    def _create_spacetime_weights_r_style(w, spt_nb, n_times, n_locs, k):
        """
        Create spacetime weights list exactly matching R's spt_wt function.
        """
        # Get base spatial weights from weights object  
        wt_base = []
        for i in range(n_locs):
            region_id = w.id_order[i]
            wt_base.append(w.weights[region_id])
        
        # Create spacetime weights list
        spt_wt = []
        
        # For each spacetime neighbor list
        for st_idx, neighbors in enumerate(spt_nb):
            loc_idx = st_idx % n_locs
            
            # Number of time periods included (current + k lags)
            n_periods = min((st_idx // n_locs) + 1, k + 1)
            
            # Replicate the base weights for each time period
            weights = wt_base[loc_idx] * n_periods
            
            # Trim to match number of neighbors
            weights = weights[:len(neighbors)]
            
            spt_wt.append(weights)
        
        return spt_wt
    
    @staticmethod
    def _find_xj_r_style(values, spt_nb):
        """
        Implement R's find_xj function: get neighbor values for each observation.
        Fixed: Add bounds checking to handle incomplete spacetime cubes.
        """
        xj = []
        max_idx = len(values) - 1
        
        for neighbors in spt_nb:
            # Filter out neighbor indices that exceed the actual data size
            valid_neighbors = [idx for idx in neighbors if 0 <= idx <= max_idx]
            neighbor_values = [values[neighbor_idx] for neighbor_idx in valid_neighbors]
            xj.append(neighbor_values)
        return xj
    
    @staticmethod
    def _local_g_spt_r_exact(values, times, spt_nb, spt_wt, n_locs):
        """
        Exact implementation of R's local_g_spt function.
        """
        # Get neighbor values using find_xj equivalent
        xj = GiStar._find_xj_r_style(values, spt_nb)
        
        # Initialize result array
        all_gis = np.zeros(len(values))
        
        # Process each time period (starts at 0, step by n_locs)
        for t in range(len(times)):
            start_idx = t * n_locs
            end_idx = start_idx + n_locs
            
            # Get values for this time period
            x_time = values[start_idx:end_idx]
            xj_time = xj[start_idx:end_idx]
            wt_time = spt_wt[start_idx:end_idx]
            
            # Calculate Gi* for this time period using R's exact formula
            gi_time = GiStar._local_g_spt_calc_r_exact(x_time, xj_time, wt_time)
            
            # Store results
            all_gis[start_idx:end_idx] = gi_time
        
        return all_gis
    
    @staticmethod
    def _local_g_spt_calc_r_exact(x, xj, wj):
        """
        Exact implementation of R's local_g_spt_calc function.
        Fixed: Handle mismatched neighbor/weight lengths due to bounds checking.
        """
        n = len(x)
        
        # xibar: mean of x repeated n times
        xibar = np.full(n, np.mean(x))
        
        # lx: weighted sum of neighbors for each observation
        # Handle case where xj and wj might have different lengths due to filtered neighbors
        lx = np.zeros(n)
        for i in range(n):
            if i < len(xj) and i < len(wj):
                xj_i = np.array(xj[i]) if len(xj[i]) > 0 else np.array([])
                wj_i = np.array(wj[i][:len(xj_i)]) if len(wj[i]) > 0 else np.array([])  # Match lengths
                if len(xj_i) > 0 and len(wj_i) > 0:
                    lx[i] = np.sum(xj_i * wj_i)
        
        # si2: variance term (sum of squared deviations / n)
        si2 = np.full(n, np.sum((x - np.mean(x)) ** 2) / n)
        
        # Wi: sum of weights for each observation (only for valid lengths)
        Wi = np.zeros(n)
        S1i = np.zeros(n)
        for i in range(n):
            if i < len(wj):
                wj_i = np.array(wj[i][:len(xj[i])]) if i < len(xj) and len(wj[i]) > 0 else np.array([])
                if len(wj_i) > 0:
                    Wi[i] = np.sum(wj_i)
                    S1i[i] = np.sum(wj_i ** 2)
        
        # EG: expected value
        EG = Wi * xibar
        
        # Numerator
        numerator = lx - EG
        
        # VG: variance (R's exact formula)
        VG = si2 * ((n * S1i - Wi**2) / (n - 1))
        
        # Final Gi* calculation
        result = np.zeros(n)
        for i in range(n):
            if VG[i] > 0:
                result[i] = numerator[i] / np.sqrt(VG[i])
            else:
                result[i] = 0.0
        
        return result
    
    @staticmethod
    def _conditional_permute_neighbors_r_style(spt_nb, n_locs):
        """
        Implement R's conditional permutation of neighbors exactly:
        For each location i, randomly reassign its neighbors while keeping same neighbor count.
        This matches R's shuffle_nbs: sample(x[-i], size=card) where x=1:n
        """
        permuted_nb = []
        n_total = len(spt_nb)
        
        for i, neighbors in enumerate(spt_nb):
            card = len(neighbors)  # Number of neighbors to maintain
            
            if card > 0:
                # R's approach: sample 'card' neighbors from {0, 1, ..., n_total-1} \ {i}
                all_indices = list(range(n_total))
                all_indices.remove(i)  # Exclude self (location i)
                
                # Sample 'card' random neighbors (without replacement)
                if len(all_indices) >= card:
                    new_neighbors = np.random.choice(all_indices, size=card, replace=False).tolist()
                else:
                    # If not enough alternatives, sample with replacement
                    new_neighbors = np.random.choice(all_indices, size=card, replace=True).tolist()
                
                permuted_nb.append(new_neighbors)
            else:
                permuted_nb.append([])
        
        return permuted_nb
    
    @staticmethod
    def _calculate_p_values_r_exact(values, times, spt_nb, spt_wt, n_locs, observed_gi, nsim):
        """
        Exact implementation of R's p-value calculation using conditional permutation.
        CRITICAL FIX: R permutes NEIGHBOR STRUCTURE, not values!
        """
        n_total = len(values)
        
        # Create simulation results matrix
        sim_results = np.zeros((n_total, nsim))
        
        # Run simulations
        for sim in range(nsim):
            # CRITICAL FIX: R conditionally permutes NEIGHBOR STRUCTURE
            # Keep values fixed, change spatial relationships!
            perm_spt_nb = GiStar._conditional_permute_neighbors_r_style(spt_nb, n_locs)
            
            # Calculate Gi* for original values with permuted neighbor structure
            sim_gi = GiStar._local_g_spt_r_exact(values, times, perm_spt_nb, spt_wt, n_locs)
            sim_results[:, sim] = sim_gi
        
        # Calculate p-values using R's exact method
        p_values = np.ones(n_total)
        
        for i in range(n_total):
            if not np.isnan(observed_gi[i]):
                # R uses this exact formula for two-tailed test
                upper_tail = np.sum(sim_results[i, :] >= observed_gi[i]) + 1
                lower_tail = np.sum(sim_results[i, :] <= observed_gi[i]) + 1
                
                # Two-tailed p-value (minimum of both tails)
                p_values[i] = min(upper_tail, lower_tail) / (nsim + 1)
            else:
                p_values[i] = 1.0
        
        return p_values
    
    # Keep the old method for backward compatibility but mark as deprecated
    @staticmethod
    def calculate_gi_star_by_period(gdf, w, region_id_field, time_period_field, value):
        """
        DEPRECATED: Use calculate_gi_star_spacetime instead for R compatibility.
        This method uses period-by-period calculation which doesn't match R sfdep.
        """
        print("WARNING: This method is deprecated. Use calculate_gi_star_spacetime for R sfdep compatibility.")
        return GiStar.calculate_gi_star_spacetime(gdf, w, region_id_field, time_period_field, value)
