import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from .ehsa_classification import EhsaClassification
from .gi_star import GiStar
from .pre_processing import PreProcessing
from .spacial_weights import SpacialWeights
from .ehsa_plotting import EhsaPlotting


class EmergingHotspotAnalysis:
    @staticmethod
    def _setup_logger(save_log_path=None):
        """
        Setup logger for EHSA analysis with both console and file output.
        
        Parameters:
        -----------
        save_log_path : str or Path, optional
            Directory path where to save the log file. If None, only console logging.
            
        Returns:
        --------
        tuple
            (logger, log_file_path) where log_file_path is None if save_log_path is None
        """
        # Create a unique logger name to avoid conflicts
        logger_name = f"ehsa_analysis_{id(EmergingHotspotAnalysis)}"
        logger = logging.getLogger(logger_name)
        
        # Clear any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger.setLevel(logging.INFO)
        
        # Create formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler - always present
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        log_file_path = None
        
        # File handler - only if save_log_path is provided
        if save_log_path:
            try:
                # Create the directory if it doesn't exist
                save_dir = Path(save_log_path)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"ehsa_log_{timestamp}.txt"
                log_file_path = save_dir / log_filename
                
                # Create file handler
                file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"EHSA Analysis Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Log file: {log_file_path}")
                logger.info("=" * 70)
                
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
                log_file_path = None
        
        return logger, log_file_path

    @staticmethod
    def emerging_hotspot_analysis(
        df, region_id_field, time_period_field, value, seed=77, save_log=False, log_dir=None, k=1, nsim=49
    ):
        """
        Perform Emerging Hot Spot Analysis (EHSA) on spatio-temporal data.
        Now matches R sfdep implementation more closely.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with spatial and temporal data
        region_id_field : str
            Column name containing spatial region identifiers (e.g., 'geohash_6')
        time_period_field : str
            Column name containing time period information
        value : str
            Column name containing the values to analyze
        seed : int, default 77
            Random seed for reproducibility
        save_log : bool, default False
            Whether to save execution log to a file
        log_dir : str or Path, optional
            Directory path where to save the log file. If None and save_log=True,
            saves to current working directory. Ignored if save_log=False.
        k : int, default 1
            Number of time lags to include in neighbors (matches R sfdep default)
        nsim : int, default 49
            Number of simulations for p-value calculation (reduced from R's 199 for speed)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing EHSA results with classifications
            
        Examples:
        ---------
        >>> # Basic usage (matches R sfdep)
        >>> results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
        ...     df, 'geohash_6', 'time_period', 'cbk_count'
        ... )
        
        >>> # With R sfdep exact parameters  
        >>> results = EmergingHotspotAnalysis.emerging_hotspot_analysis(
        ...     df, 'geohash_6', 'time_period', 'cbk_count',
        ...     k=1, nsim=199  # Matches R defaults exactly
        ... )
        """
        start_time = time.time()
        
        # Setup logging
        if save_log:
            log_path = log_dir if log_dir is not None else '.'  # Use current directory if no log_dir specified
        else:
            log_path = None
        logger, log_file_path = EmergingHotspotAnalysis._setup_logger(log_path)
        
        logger.info("🚀 Starting Emerging Hotspot Analysis")
        logger.info(f"📊 Input DataFrame shape: {df.shape}")
        logger.info(f"🎯 Analysis parameters:")
        logger.info(f"   - Region ID field: {region_id_field}")
        logger.info(f"   - Time period field: {time_period_field}")  
        logger.info(f"   - Value field: {value}")
        logger.info(f"   - Random seed: {seed}")
        logger.info(f"   - Time lags (k): {k}")
        logger.info(f"   - Simulations (nsim): {nsim}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Data validation summary
        logger.info(f"📈 Data overview:")
        logger.info(f"   - Total rows: {len(df):,}")
        logger.info(f"   - Unique regions: {df[region_id_field].nunique():,}")
        logger.info(f"   - Unique time periods: {df[time_period_field].nunique()}")
        logger.info(f"   - Value range: [{df[value].min():.3f}, {df[value].max():.3f}]")
        logger.info(f"   - Missing values in {value}: {df[value].isnull().sum():,}")

        # Step 1: Data validation and cleaning
        logger.info("🔧 Step 1: Validating and cleaning input data...")
        df_clean = PreProcessing.validate_and_clean_data(
            df, value, region_id_field
        )
        logger.info(f"✅ Data validation completed")

        # Step 2: Preprocessing
        logger.info("🗺️  Step 2: Processing spatial data...")
        gdf = PreProcessing.process(df_clean)
        logger.info(f"✅ Preprocessing completed: {gdf.shape}")
        logger.info(f"   - GeoDataFrame created with {len(gdf)} records")
        
        # Step 2.5: Complete spacetime cube (matching R sfdep)
        logger.info("🔲 Step 2.5: Creating complete spacetime cube...")
        gdf_complete = PreProcessing.complete_spacetime_cube(gdf, region_id_field, time_period_field)
        logger.info(f"✅ Complete spacetime cube created: {gdf_complete.shape}")
        
        # Filter out rows with NaN values (matching R: filter(!is.na(value)))
        before_filter = len(gdf_complete)
        gdf_complete = gdf_complete.dropna(subset=[value])
        after_filter = len(gdf_complete)
        logger.info(f"🔍 Filtered out {before_filter - after_filter} rows with NaN values in '{value}'")
        logger.info(f"   - Final dataset for analysis: {gdf_complete.shape}")
        
        # CRITICAL: Sort by time_period then region_id (matching R's spacetime cube order)
        logger.info("🔄 Sorting data to match R spacetime cube order...")
        gdf_complete = gdf_complete.sort_values([time_period_field, region_id_field]).reset_index(drop=True)
        logger.info(f"   - Data sorted by [{time_period_field}, {region_id_field}]")
        
        # Use the complete cube for analysis
        gdf = gdf_complete
        
        # Step 3: Spatial weights calculation
        logger.info("🔗 Step 3: Calculating spatial weights...")
        step3_start = time.time()
        w = SpacialWeights.calculate_spatial_weights(gdf, region_id_field)
        step3_time = time.time() - step3_start
        logger.info(f"✅ Spatial weights calculated in {step3_time:.2f}s")
        logger.info(f"   - Spatial weights matrix: {w.n} regions")
        logger.info(f"   - Average neighbors per region: {w.mean_neighbors:.1f}")
        
        # Step 4: Gi* statistics calculation (using R sfdep compatible method)
        logger.info("📊 Step 4: Calculating Getis-Ord Gi* statistics (spacetime method)...")
        step4_start = time.time()
        # Use the new spacetime method that matches R sfdep implementation
        gdf = GiStar.calculate_gi_star_spacetime(
            gdf, w, region_id_field, time_period_field, value, k=k, nsim=nsim
        )
        step4_time = time.time() - step4_start
        logger.info(f"✅ Gi* statistics calculated in {step4_time:.2f}s using spacetime method")
        logger.info(f"   - Output shape: {gdf.shape}")
        logger.info(f"   - Used {nsim} simulations for p-values")
        
        # Step 5: EHSA classification
        logger.info("🎯 Step 5: Performing emerging hotspot classification...")
        step5_start = time.time()
        results = EhsaClassification.perform_emerging_hotspot_analysis(
            gdf, w, region_id_field, time_period_field, original_value_field=value
        )
        step5_time = time.time() - step5_start
        logger.info(f"✅ EHSA classification completed in {step5_time:.2f}s")
        logger.info(f"   - Results shape: {results.shape}")

        # Results summary
        elapsed_time = time.time() - start_time
        elapsed_formatted = str(timedelta(seconds=elapsed_time))
        
        logger.info("=" * 50)
        logger.info("📋 ANALYSIS RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total execution time: {elapsed_formatted}")
        
        # Classification distribution
        if 'classification' in results.columns:
            classification_counts = results['classification'].value_counts()
            logger.info(f"🏷️  Classification distribution:")
            for classification, count in classification_counts.items():
                percentage = (count / len(results)) * 100
                logger.info(f"   - {classification}: {count:,} regions ({percentage:.1f}%)")
        
        # Statistical summary
        if 'tau' in results.columns:
            tau_stats = results['tau'].describe()
            logger.info(f"📊 Mann-Kendall Tau statistics:")
            logger.info(f"   - Mean: {tau_stats['mean']:.4f}")
            logger.info(f"   - Std: {tau_stats['std']:.4f}")
            logger.info(f"   - Range: [{tau_stats['min']:.4f}, {tau_stats['max']:.4f}]")
        
        logger.info("=" * 50)
        logger.info("✅ EHSA Analysis completed successfully!")
        
        if log_file_path:
            logger.info(f"📄 Detailed log saved to: {log_file_path}")
            
        # Clean up logger handlers to prevent memory leaks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        return results

    @staticmethod
    def launch_visualization(ehsa_results_df=None):
        """
        Launch the interactive EHSA visualization tool in the browser.
        
        Parameters:
        -----------
        ehsa_results_df : pandas.DataFrame, optional
            EHSA results DataFrame to visualize. If None, opens empty tool.
            
        Returns:
        --------
        str
            Path to the opened HTML file
            
        Example:
        --------
        >>> from pyehsa.emerging_hotspot_analysis import EmergingHotspotAnalysis
        >>> # Run analysis
        >>> results = EmergingHotspotAnalysis.emerging_hotspot_analysis(df, 'geohash_6', 'time_period', 'cbk_count')
        >>> # Launch visualization with results
        >>> EmergingHotspotAnalysis.launch_visualization(results)
        """
        return EhsaPlotting.open_visualization_tool(ehsa_results_df)

    @staticmethod
    def save_visualization(file_path, ehsa_results_df=None, include_data=True):
        """
        Save the interactive EHSA visualization tool to an HTML file.
        Similar to folium.save() or plotly.write_html() - saves without opening browser.
        
        Parameters:
        -----------
        file_path : str or Path
            Path where to save the HTML file (e.g., 'my_ehsa_visualization.html')
        ehsa_results_df : pandas.DataFrame, optional
            EHSA results DataFrame to embed in the visualization.
        include_data : bool, default True
            Whether to embed the data directly in the HTML file.
            
        Returns:
        --------
        str
            Absolute path to the saved HTML file
            
        Example:
        --------
        >>> from pyehsa.emerging_hotspot_analysis import EmergingHotspotAnalysis
        >>> # Run analysis and save visualization
        >>> results = EmergingHotspotAnalysis.emerging_hotspot_analysis(df, 'geohash_6', 'time_period', 'cbk_count')
        >>> EmergingHotspotAnalysis.save_visualization('ehsa_results.html', results)
        """
        return EhsaPlotting.save_ehsa_visualization_tool(file_path, ehsa_results_df, include_data)
