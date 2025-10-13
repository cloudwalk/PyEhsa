import numpy as np
import pandas as pd

from .mann_kendall import MannKendall


class EhsaClassification:

    @staticmethod
    def new_hotspot(gs, sigs, n):
        return (np.sum(sigs) == 1) and sigs[n - 1] and gs[n - 1] > 0

    @staticmethod
    def consecutive_hotspot(gs, sigs, n):
        # R: if (!(sigs[n] && gs[n] > 0)) return(FALSE)
        if not (sigs[n - 1] and gs[n - 1] > 0):
            return False

        # R's complex final run calculation: which(diff(cumsum(c(rev(sigs), FALSE))) == 0)[1]
        # This finds the length of the final consecutive run of TRUE values
        reversed_sigs = np.flip(sigs)
        extended_sigs = np.append(reversed_sigs, False)
        cumsum_vals = np.cumsum(extended_sigs)
        diff_vals = np.diff(cumsum_vals)
        
        # Find first zero in diff (indicates end of consecutive run)
        zero_indices = np.where(diff_vals == 0)[0]
        if len(zero_indices) == 0:
            n_final_run = n  # All periods are consecutive
        else:
            n_final_run = zero_indices[0] + 1
        
        # R: run_index <- seq(n - n_final_run + 1, n, by = 1)
        run_start = n - n_final_run
        run_index = np.arange(run_start, n)
        
        # Create boolean arrays for before and during run
        before_run = np.arange(n) < run_start
        in_run = np.arange(n) >= run_start
        
        # R conditions:
        # all(!sigs[-run_index]) && all(sigs[run_index]) && 
        # all(!((sigs[-run_index]) & (gs[-run_index] > 0))) &&
        # all(!sigs[-run_index]) && (sum(sigs & gs > 0) / n < 0.9)
        
        return (
            (np.all(~sigs[before_run]) if run_start > 0 else True) and  # No significant periods before run
            np.all(sigs[in_run]) and  # All periods in run are significant
            np.sum(sigs & (gs > 0)) / n < 0.9  # Less than 90% are significant hotspots overall
        )

    @staticmethod
    def intensifying_hotspot(gs, sigs, n, tau, tau_p, threshold):
        # R: (sum(sigs) / n >= .9) && (sigs[n]) && (tau > 0) && (tau_p < threshold)
        # CRITICAL: R uses sum(sigs) not sum(sigs & (gs > 0))!
        return (
            (np.sum(sigs) / n >= 0.9)  # 90% of periods are significant (any direction)
            and sigs[n - 1]  # Last period is significant  
            and tau > 0  # Increasing trend
            and tau_p < threshold  # Significant trend
        )

    @staticmethod
    def persistent_hotspot(gs, sigs, tau_p, threshold, n):
        return (np.sum(sigs & (gs > 0)) / n >= 0.9) and tau_p > threshold

    def diminishing_hotspot(gs, sigs, tau, tau_p, threshold, n):
        return (
            (np.sum(sigs & (gs > 0)) / n >= 0.9)
            and (sigs[n - 1] and gs[n - 1] > 0)
            and tau < 0
            and tau_p <= threshold
        )

    @staticmethod
    def sporadic_hotspot(gs, sigs, n):
        return (
            (np.sum(sigs & (gs > 0)) / n < 0.9)
            and not np.any(sigs & (gs < 0))
            and np.any(sigs & (gs > 0))
        )

    @staticmethod
    def sporadic_coldspot(gs, sigs, n):
        return (
            (np.sum(sigs & (gs < 0)) / n < 0.9)
            and not np.any(sigs & (gs > 0))
            and np.any(sigs & (gs < 0))
        )

    @staticmethod
    def oscillating_hotspot(gs, sigs, n):
        return (
            (np.sum(sigs & (gs > 0)) / n <= 0.9)
            and (gs[n - 1] > 0 and sigs[n - 1])
            and np.any(sigs & (gs < 0))
        )

    @staticmethod
    def historical_hotspot(gs, sigs, n):
        return (np.sum(sigs & (gs > 0)) / n >= 0.9) and (gs[n - 1] < 0)

    @staticmethod
    def new_coldspot(gs, sigs, n):
        return (np.sum(sigs) == 1) and sigs[n - 1] and gs[n - 1] < 0

    @staticmethod
    def consecutive_coldspot(gs, sigs, n):
        # R: if (!(sigs[n] && gs[n] < 0)) return(FALSE)  
        if not (sigs[n - 1] and gs[n - 1] < 0):
            return False

        # R's complex final run calculation (same as hotspot)
        reversed_sigs = np.flip(sigs)
        extended_sigs = np.append(reversed_sigs, False)
        cumsum_vals = np.cumsum(extended_sigs)
        diff_vals = np.diff(cumsum_vals)
        
        zero_indices = np.where(diff_vals == 0)[0]
        if len(zero_indices) == 0:
            n_final_run = n
        else:
            n_final_run = zero_indices[0] + 1
        
        run_start = n - n_final_run
        before_run = np.arange(n) < run_start
        in_run = np.arange(n) >= run_start
        
        return (
            (np.all(~sigs[before_run]) if run_start > 0 else True) and
            np.all(sigs[in_run]) and
            np.sum(sigs & (gs < 0)) / n < 0.9
        )

    @staticmethod
    def intensifying_coldspot(gs, sigs, n, tau, tau_p, threshold):
        # R: (sum(sigs & gs < 0) / n >= .9) && (sigs[n]) && (tau < 0) && (tau_p <= threshold)
        return (
            (np.sum(sigs & (gs < 0)) / n >= 0.9)  # 90% of periods are significant coldspots
            and sigs[n - 1]  # Last period is significant
            and tau < 0  # Decreasing trend (intensifying coldspot)
            and tau_p <= threshold  # Significant trend
        )

    @staticmethod
    def persistent_coldspot(gs, sigs, n, tau_p, threshold):
        return (np.sum(sigs & (gs < 0)) / n >= 0.9) and tau_p > threshold

    @staticmethod
    def diminishing_coldspot(gs, sigs, n, tau, tau_p, threshold):
        return (
            (np.sum(sigs & (gs < 0)) / n >= 0.9)
            and sigs[n - 1]
            and tau > 0
            and tau_p <= threshold
        )

    @staticmethod
    def oscillating_coldspot(gs, sigs, n):
        return (
            (np.sum(sigs & (gs < 0)) / n < 0.9)
            and (gs[n - 1] < 0 and sigs[n - 1])
            and np.any(sigs & (gs > 0))
        )

    @staticmethod
    def historical_coldspot(gs, sigs, n):
        return (gs[n - 1] > 0) and (np.sum(sigs & (gs < 0)) / n >= 0.9)

    @staticmethod
    def classify_hotspot(gs, sigs, tau, tau_p, threshold=0.01):
        """Classify a location based on its Gi* statistics time series and Mann-Kendall results,
        and provide a detailed explanation."""
        n = len(gs)
        gs_arr = np.array(gs)
        sigs_arr = np.array(sigs, dtype=bool)

        explanation_details = {}

        # Helper to create a base for explanation values
        def get_base_values():
            return {
                "gi_star_series": [round(g, 4) for g in gs_arr.tolist()],
                "significance_series": sigs_arr.tolist(),
                "num_time_periods": n,
                "mann_kendall_tau": round(tau, 4) if not np.isnan(tau) else None,
                "mann_kendall_p_value": round(tau_p, 4) if not np.isnan(tau_p) else None,
                "mk_significance_threshold": threshold,
            }

        if EhsaClassification.new_hotspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Exactly one significant period: {np.sum(sigs_arr) == 1} (actual sum: {np.sum(sigs_arr)})",
                    f"The last period ({n-1}) is significant: {sigs_arr[n-1]}",
                    f"The last period ({n-1}) Gi* > 0: {gs_arr[n-1] > 0} (actual Gi*: {gs_arr[n-1]:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "new hotspot", "explanation": explanation_details}

        if EhsaClassification.new_coldspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Exactly one significant period: {np.sum(sigs_arr) == 1} (actual sum: {np.sum(sigs_arr)})",
                    f"The last period ({n-1}) is significant: {sigs_arr[n-1]}",
                    f"The last period ({n-1}) Gi* < 0: {gs_arr[n-1] < 0} (actual Gi*: {gs_arr[n-1]:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "new coldspot", "explanation": explanation_details}

        if EhsaClassification.consecutive_hotspot(gs_arr, sigs_arr, n):
            is_last_period_hotspot = sigs_arr[n-1] and gs_arr[n-1] > 0
            n_final_run = 0
            if is_last_period_hotspot:
                n_final_run = 1
                for i in range(n - 2, -1, -1):
                    if not (sigs_arr[i] and gs_arr[i] > 0): break
                    n_final_run += 1
            run_start = n - n_final_run
            prior_sigs = sigs_arr[:run_start]
            run_sigs = sigs_arr[run_start:]
            explanation_details = {
                "criteria_met": [
                    f"Last period ({n-1}) is a significant hotspot: {is_last_period_hotspot} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"All periods before the current run of hotspots are non-significant: {np.all(~prior_sigs) if n_final_run > 0 and run_start > 0 else 'N/A (no prior periods or no run)'}",
                    f"All periods in the final run are significant hotspots: {np.all(run_sigs) if n_final_run > 0 else 'N/A (no run)'}",
                    f"Less than 90% of all periods are significant hotspots: {np.sum(sigs_arr & (gs_arr > 0)) / n < 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"Length of final run of hotspots: {n_final_run}"
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "consecutive hotspot", "explanation": explanation_details}

        if EhsaClassification.consecutive_coldspot(gs_arr, sigs_arr, n):
            is_last_period_coldspot = sigs_arr[n-1] and gs_arr[n-1] < 0
            n_final_run = 0
            if is_last_period_coldspot:
                n_final_run = 1
                for i in range(n - 2, -1, -1):
                    if not (sigs_arr[i] and gs_arr[i] < 0): break
                    n_final_run += 1
            run_start = n - n_final_run
            prior_sigs = sigs_arr[:run_start]
            run_sigs = sigs_arr[run_start:]
            explanation_details = {
                "criteria_met": [
                    f"Last period ({n-1}) is a significant coldspot: {is_last_period_coldspot} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"All periods before the current run of coldspots are non-significant: {np.all(~prior_sigs) if n_final_run > 0 and run_start > 0 else 'N/A (no prior periods or no run)'}",
                    f"All periods in the final run are significant coldspots: {np.all(run_sigs) if n_final_run > 0 else 'N/A (no run)'}",
                    f"Less than 90% of all periods are significant coldspots: {np.sum(sigs_arr & (gs_arr < 0)) / n < 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                    f"Length of final run of coldspots: {n_final_run}"
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "consecutive coldspot", "explanation": explanation_details}

        if EhsaClassification.intensifying_hotspot(gs_arr, sigs_arr, n, tau, tau_p, threshold):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant hotspots: {np.sum(sigs_arr & (gs_arr > 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"The last period ({n-1}) is a significant hotspot: {sigs_arr[n-1] and gs_arr[n-1] > 0} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"Mann-Kendall tau > 0 (increasing trend): {tau > 0} (actual tau: {tau:.4f})",
                    f"Mann-Kendall p-value < threshold ({threshold}): {tau_p < threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "intensifying hotspot", "explanation": explanation_details}

        if EhsaClassification.intensifying_coldspot(gs_arr, sigs_arr, n, tau, tau_p, threshold):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant coldspots: {np.sum(sigs_arr & (gs_arr < 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                    f"The last period ({n-1}) is significant (coldspot implied by overall <0): {sigs_arr[n-1]} (Gi*: {gs_arr[n-1]:.4f})", # Original check was just sigs[n-1]
                    f"The last period ({n-1}) is a significant coldspot: {sigs_arr[n-1] and gs_arr[n-1] < 0}", # Adding explicit check for clarity
                    f"Mann-Kendall tau < 0 (intensifying coldspot trend): {tau < 0} (actual tau: {tau:.4f})",
                    f"Mann-Kendall p-value <= threshold ({threshold}): {tau_p <= threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "intensifying coldspot", "explanation": explanation_details}

        if EhsaClassification.persistent_hotspot(gs_arr, sigs_arr, tau_p, threshold, n):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant hotspots: {np.sum(sigs_arr & (gs_arr > 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"Mann-Kendall p-value > threshold ({threshold}) (no significant trend): {tau_p > threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "persistent hotspot", "explanation": explanation_details}

        if EhsaClassification.persistent_coldspot(gs_arr, sigs_arr, n, tau_p, threshold):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant coldspots: {np.sum(sigs_arr & (gs_arr < 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                    f"Mann-Kendall p-value > threshold ({threshold}) (no significant trend): {tau_p > threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "persistent coldspot", "explanation": explanation_details}

        if EhsaClassification.diminishing_hotspot(gs_arr, sigs_arr, tau, tau_p, threshold, n):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant hotspots: {np.sum(sigs_arr & (gs_arr > 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"The last period ({n-1}) is a significant hotspot: {sigs_arr[n-1] and gs_arr[n-1] > 0} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"Mann-Kendall tau < 0 (decreasing trend): {tau < 0} (actual tau: {tau:.4f})",
                    f"Mann-Kendall p-value <= threshold ({threshold}): {tau_p <= threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "diminishing hotspot", "explanation": explanation_details}

        if EhsaClassification.diminishing_coldspot(gs_arr, sigs_arr, n, tau, tau_p, threshold):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods are significant coldspots: {np.sum(sigs_arr & (gs_arr < 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                    f"The last period ({n-1}) is significant (coldspot implied by overall <0): {sigs_arr[n-1]} (Gi*: {gs_arr[n-1]:.4f})", # Original check
                    f"The last period ({n-1}) is a significant coldspot: {sigs_arr[n-1] and gs_arr[n-1] < 0}", # Adding explicit check
                    f"Mann-Kendall tau > 0 (diminishing coldspot trend): {tau > 0} (actual tau: {tau:.4f})",
                    f"Mann-Kendall p-value <= threshold ({threshold}): {tau_p <= threshold} (actual p-value: {tau_p:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "diminishing coldspot", "explanation": explanation_details}

        if EhsaClassification.oscillating_hotspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Proportion of significant hotspots <= 0.9: {np.sum(sigs_arr & (gs_arr > 0)) / n <= 0.9} (actual: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"Last period ({n-1}) is a significant hotspot: {gs_arr[n-1] > 0 and sigs_arr[n-1]} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"At least one significant coldspot period exists: {np.any(sigs_arr & (gs_arr < 0))}",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "oscilating hotspot", "explanation": explanation_details}

        if EhsaClassification.oscillating_coldspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Proportion of significant coldspots < 0.9: {np.sum(sigs_arr & (gs_arr < 0)) / n < 0.9} (actual: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})", # original was < 0.9
                    f"Last period ({n-1}) is a significant coldspot: {gs_arr[n-1] < 0 and sigs_arr[n-1]} (Gi*: {gs_arr[n-1]:.4f}, Sig: {sigs_arr[n-1]})",
                    f"At least one significant hotspot period exists: {np.any(sigs_arr & (gs_arr > 0))}",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "oscilating coldspot", "explanation": explanation_details}

        if EhsaClassification.historical_hotspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"At least 90% of periods were significant hotspots: {np.sum(sigs_arr & (gs_arr > 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"The last period ({n-1}) Gi* < 0 (is now cold or neutral): {gs_arr[n-1] < 0} (actual Gi*: {gs_arr[n-1]:.4f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "historical hotspot", "explanation": explanation_details}

        if EhsaClassification.historical_coldspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"The last period ({n-1}) Gi* > 0 (is now hot or neutral): {gs_arr[n-1] > 0} (actual Gi*: {gs_arr[n-1]:.4f})",
                    f"At least 90% of periods were significant coldspots: {np.sum(sigs_arr & (gs_arr < 0)) / n >= 0.9} (actual proportion: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "historical coldspot", "explanation": explanation_details}

        if EhsaClassification.sporadic_hotspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Proportion of significant hotspots < 0.9: {np.sum(sigs_arr & (gs_arr > 0)) / n < 0.9} (actual: {np.sum(sigs_arr & (gs_arr > 0)) / n:.2f})",
                    f"No significant coldspot periods exist: {not np.any(sigs_arr & (gs_arr < 0))}",
                    f"At least one significant hotspot period exists: {np.any(sigs_arr & (gs_arr > 0))}",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "sporadic hotspot", "explanation": explanation_details}

        if EhsaClassification.sporadic_coldspot(gs_arr, sigs_arr, n):
            explanation_details = {
                "criteria_met": [
                    f"Proportion of significant coldspots < 0.9: {np.sum(sigs_arr & (gs_arr < 0)) / n < 0.9} (actual: {np.sum(sigs_arr & (gs_arr < 0)) / n:.2f})",
                    f"No significant hotspot periods exist: {not np.any(sigs_arr & (gs_arr > 0))}",
                    f"At least one significant coldspot period exists: {np.any(sigs_arr & (gs_arr < 0))}",
                ],
                "values_at_decision": get_base_values(),
            }
            return {"name": "sporadic coldspot", "explanation": explanation_details}

        return {
            "name": "no pattern detected",
            "explanation": {
                "reason": "Conditions for specific hotspot/coldspot patterns were not met.",
                "values_at_decision": get_base_values(),
            },
        }

    @staticmethod
    def perform_emerging_hotspot_analysis(
        gdf, w, region_id_field, time_period_field, threshold=0.01, original_value_field=None
    ):
        results = []
        total_regions = len(gdf[region_id_field].unique())
        print(f"Total regions to process: {total_regions}")
        
        # Counter for detailed debugging (limit output)
        debug_count = 0
        max_debug = 15  # Only show first 15 problem cases

        for region, group in gdf.groupby(region_id_field):
            group = group.sort_values(time_period_field)

            gi_stats_raw = group["gi"].values
            signif_flags = group["p_sim"].values <= threshold  # R uses <= threshold, threshold=0.01
            
            # Create location_data with time series information
            location_data = []
            for _, row in group.iterrows():
                location_data.append({
                    "time_period": str(row[time_period_field]),  # Convert to string for HTML compatibility
                    "value": float(row["gi"]),
                    "is_significant": bool(row["p_sim"] <= threshold),
                    "original_value": float(row.get(original_value_field, None)) if original_value_field and pd.notna(row.get(original_value_field, None)) else None
                })

            # Keep minimal debugging for pattern analysis
            
            # Skip if insufficient data
            if len(gi_stats_raw) <= 1:
                print(f"Skipping region {region} due to insufficient time periods")
                
                # Still calculate spatial context for visualization purposes
                neighbors_list = w.neighbors.get(region, [])
                weights_list = w.weights.get(region, [])
                spatial_context_summary = {
                    "neighbors_config": neighbors_list,
                    "weights_config": weights_list,
                    "note": "Neighbor configuration available but region skipped due to insufficient time periods.",
                }
                
                # Add a record indicating skipped region if desired, or just continue
                results.append({
                    region_id_field: region,
                    "classification": "skipped_insufficient_data",
                    "classification_details": {"reason": "Less than 2 time periods."},
                    "mann_kendall_details": None,
                    "spatial_context_summary": spatial_context_summary,
                    "location_data": location_data,  # Add the time series data even for skipped regions
                    "tau": np.nan,
                    "p_value": np.nan,
                })
                continue

            # Skip if all values are NaN
            if np.all(np.isnan(gi_stats_raw)):
                print(f"Skipping region {region} due to all NaN values")
                
                # Still calculate spatial context for visualization purposes
                neighbors_list = w.neighbors.get(region, [])
                weights_list = w.weights.get(region, [])
                spatial_context_summary = {
                    "neighbors_config": neighbors_list,
                    "weights_config": weights_list,
                    "note": "Neighbor configuration available but region skipped due to all NaN values.",
                }
                
                results.append({
                    region_id_field: region,
                    "classification": "skipped_all_nan",
                    "classification_details": {"reason": "All Gi* values are NaN."},
                    "mann_kendall_details": None,
                    "spatial_context_summary": spatial_context_summary,
                    "location_data": location_data,  # Add the time series data even for skipped regions
                    "tau": np.nan,
                    "p_value": np.nan,
                })
                continue

            # Replace any remaining NaN values with 0 for MK test and classification logic
            # The raw NaNs might be important context for the user, so we note this step.
            gi_stats_processed = np.nan_to_num(gi_stats_raw, nan=0.0)


            # Spatial Context
            # Taking from the first time period of the group.
            # Assuming 'neighbors' and 'weights' columns exist from SpacialWeights step.
            
            # Get neighbors from the global PySAL weights object
            neighbors_list = w.neighbors.get(region, [])
            weights_list = w.weights.get(region, [])

            spatial_context_summary = {
                "neighbors_config": neighbors_list,
                "weights_config": weights_list,
                "note": "Neighbor and weight configuration shown is from the global weights matrix.",
            }

            mk_result_data = {"tau": np.nan, "sl": np.nan, "S": np.nan, "D": np.nan, "varS": np.nan} # Initialize
            mann_kendall_details_dict = {}

            try:
                mk_result_actual = MannKendall.mann_kendall_test(gi_stats_processed)
                # Ensure all expected keys are present, even if NaN
                for key in mk_result_data:
                    if key in mk_result_actual:
                        mk_result_data[key] = mk_result_actual[key]


                mann_kendall_details_dict = {
                    "inputs": {
                        "time_periods_count": len(gi_stats_processed),
                        "gi_star_values_over_time (processed for MK)": [round(g, 4) for g in gi_stats_processed.tolist()],
                        "original_gi_star_values (before nan_to_num)": [round(g, 4) if pd.notna(g) else None for g in gi_stats_raw.tolist()],
                    },
                    "results": {k: (round(v, 4) if isinstance(v, float) and pd.notna(v) else v) for k, v in mk_result_data.items()},
                }

                classification_info = {"name": "no pattern detected", 
                                       "explanation": {"reason": "Mann-Kendall test yielded NaN results, cannot classify.",
                                                       "values_at_decision": {
                                                            "gi_star_series": [round(g, 4) for g in gi_stats_processed.tolist()],
                                                            "significance_series": signif_flags.tolist(),
                                                            "num_time_periods": len(gi_stats_processed),
                                                            "mann_kendall_tau": mk_result_data["tau"],
                                                            "mann_kendall_p_value": mk_result_data["sl"],
                                                            "mk_significance_threshold": threshold,
                                                       }}}
                if not np.isnan(mk_result_data["tau"]) and not np.isnan(mk_result_data["sl"]):
                    classification_info = EhsaClassification.classify_hotspot(
                        gi_stats_processed, signif_flags, mk_result_data["tau"], mk_result_data["sl"], threshold
                    )
                    
                    # DEEP DEBUG: Show detailed classification logic for problem patterns
                    classification_name = classification_info["name"]
                    
                    # Track successful classifications for verification
                    if classification_name != "no pattern detected":
                        print(f"✓ {region}: {classification_name}")
                        
                    # Count no-pattern regions for analysis
                    if classification_name == "no pattern detected":
                        total_sig = np.sum(signif_flags)
                        if debug_count < 3:  # Just show first 3
                            print(f"   No pattern #{debug_count+1}: {region}, Gi* range [{np.min(gi_stats_processed):.2f}, {np.max(gi_stats_processed):.2f}], {total_sig}/10 significant")
                            debug_count += 1
                
                current_result = {
                    region_id_field: region,
                    "classification": classification_info["name"],
                    "classification_details": classification_info["explanation"],
                    "mann_kendall_details": mann_kendall_details_dict,
                    "spatial_context_summary": spatial_context_summary,
                    "location_data": location_data,  # Add the time series data
                    "tau": mk_result_data["tau"], # Keep top-level for convenience
                    "p_value": mk_result_data["sl"], # Keep top-level for convenience
                }
                results.append(current_result)

            except Exception as e:
                print(f"Error processing region {region}: {str(e)}")
                
                # Ensure spatial context is available even for error cases
                neighbors_list = w.neighbors.get(region, [])
                weights_list = w.weights.get(region, [])
                error_spatial_context = {
                    "neighbors_config": neighbors_list,
                    "weights_config": weights_list,
                    "note": "Neighbor configuration available but region had processing errors.",
                }
                
                results.append(
                    {
                        region_id_field: region,
                        "classification": "error_processing",
                        "classification_details": {"error_message": str(e)},
                        "mann_kendall_details": mann_kendall_details_dict, # Include what we have
                        "spatial_context_summary": error_spatial_context, # Ensure spatial context is available
                        "location_data": location_data,  # Add the time series data even for error cases
                        "tau": mk_result_data.get("tau", np.nan),
                        "p_value": mk_result_data.get("sl", np.nan),
                    }
                )
                continue

        results_df = pd.DataFrame(results)
        print(f"Total results processed: {len(results_df)}")
        return results_df
