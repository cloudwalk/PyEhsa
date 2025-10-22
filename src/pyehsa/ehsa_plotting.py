import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from shapely import wkt
import folium
import webbrowser
import os
import tempfile
import shutil
import json
from pathlib import Path


class EhsaPlotting:
    @staticmethod
    def _sanitize_csv_value(value):
        """
        Sanitize a value to prevent CSV formula injection attacks.
        
        Prefixes potentially dangerous characters that could be interpreted
        as formulas by spreadsheet applications (Excel, LibreOffice, etc).
        
        Parameters:
        -----------
        value : any
            Value to sanitize
            
        Returns:
        --------
        any
            Sanitized value safe for CSV export
        """
        if pd.isna(value) or value is None:
            return value
            
        # Convert to string for checking
        str_value = str(value)
        
        # Check if value starts with dangerous characters
        dangerous_chars = ('=', '+', '-', '@', '\t', '\r', '\n', '|')
        
        if str_value and str_value[0] in dangerous_chars:
            # Prefix with single quote to prevent formula execution
            # This is the standard CSV injection mitigation technique
            return "'" + str_value
        
        return value
    
    @staticmethod
    def _sanitize_dataframe_for_csv(df):
        """
        Sanitize all columns in a DataFrame to prevent CSV formula injection.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to sanitize
            
        Returns:
        --------
        pandas.DataFrame
            Sanitized DataFrame safe for CSV export
        """
        df_sanitized = df.copy()
        
        for col in df_sanitized.columns:
            df_sanitized[col] = df_sanitized[col].apply(EhsaPlotting._sanitize_csv_value)
        
        return df_sanitized
    
    @staticmethod
    def plot_ehsa_map_static(df, title="Emerging Hotspots"):
        # Define color scheme
        hotspot_colors = {
            "no pattern detected": "lightgray",
            "new hotspot": "mediumturquoise",
            "new coldspot": "#87CEEB",  # Light sky blue
            "sporadic hotspot": "lightsalmon",
            "sporadic coldspot": "#B0E0E6",  # Powder blue
            "consecutive hotspot": "orangered",
            "consecutive coldspot": "#4682B4",  # Steel blue
            "persistent hotspot": "saddlebrown",
            "persistent coldspot": "#000080",  # Navy
            "intensifying hotspot": "purple",
            "intensifying coldspot": "#483D8B",  # Dark slate blue
            "diminishing hotspot": "orange",
            "diminishing coldspot": "#1E90FF",  # Dodger blue
            "historical hotspot": "#8B4513",  # Saddle brown
            "historical coldspot": "#00008B",  # Dark blue
            "oscilating hotspot": "#FF8C00",  # Dark orange
            "oscilating coldspot": "#0000CD",  # Medium blue
        }

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(25, 15))

        # Convert to GeoDataFrame if not already
        gdf = gpd.GeoDataFrame(df, geometry="geometry")

        # Plot each classification category
        for classification, color in hotspot_colors.items():
            mask = gdf["classification"] == classification
            if mask.any():  # Only plot if there are areas with this classification
                gdf[mask].plot(ax=ax, color=color, alpha=0.7, label=classification)

        # Customize the plot
        ax.set_title(title, fontsize=16, pad=20)
        ax.axis("equal")
        ax.set_xticks([])  # Remove x axis ticks
        ax.set_yticks([])  # Remove y axis ticks

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        # Sort legend items to group hot/cold spots
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        # Place legend outside of the plot
        ax.legend(
            handles,
            labels,
            title="Classification",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_ehsa_map_interactive(
        df, region_id_field, title="Emerging Hotspot Analysis", lib="plotly"
    ):
        # Convert to GeoDataFrame if not already (WKT parsing)
        if df["geometry"].dtype == "object" and isinstance(df["geometry"].iloc[0], str):
            try:
                df["geometry"] = df["geometry"].apply(wkt.loads)
            except Exception as e:
                raise ValueError(f"Failed to parse WKT geometry strings: {str(e)}")

        # Convert to GeoDataFrame if not already
        try:
            gdf = gpd.GeoDataFrame(df, geometry="geometry")
        except Exception as e:
            raise ValueError(f"Failed to create GeoDataFrame: {str(e)}")

        # Handle empty GeoDataFrame
        if gdf.empty:
            print(f"Warning: GeoDataFrame is empty. Cannot generate map for '{title}'.")
            if lib == "plotly":
                return px.choropleth_mapbox(
                    title=f"{title} (No data)",
                    mapbox_style="carto-positron",
                    center={"lat": 0, "lon": 0},
                    zoom=1
                )
            elif lib == "folium":
                empty_folium_map = folium.Map(location=[0, 0], zoom_start=1, tiles="CartoDB positron")
                # Optionally add a title to the empty Folium map
                title_html = f'<h3 align="center" style="font-size:16px; font-family: Arial, sans-serif;"><b>{title} (No data)</b></h3>'
                empty_folium_map.get_root().html.add_child(folium.Element(title_html))
                return empty_folium_map
            else:
                raise ValueError("Invalid 'lib' argument. Supported values are 'plotly' or 'folium'.")

        # Create custom hover text
        gdf["hover_text"] = (
            f"Region ID: "
            + gdf[region_id_field].astype(str)
            + "<br>"
            + "Classification: "
            + gdf["classification"]
            + "<br>"
            + "Tau: "
            + gdf["tau"].round(4).astype(str)
            + "<br>"
            + "P-value: "
            + gdf["p_value"].round(4).astype(str)
        )

        # Define color scheme
        hotspot_colors = {
            "no pattern detected": "lightgray",
            "new hotspot": "mediumturquoise",
            "new coldspot": "#87CEEB",  # Light sky blue
            "sporadic hotspot": "lightsalmon",
            "sporadic coldspot": "#B0E0E6",  # Powder blue
            "consecutive hotspot": "orangered",
            "consecutive coldspot": "#4682B4",  # Steel blue
            "persistent hotspot": "saddlebrown",
            "persistent coldspot": "#000080",  # Navy
            "intensifying hotspot": "purple",
            "intensifying coldspot": "#483D8B",  # Dark slate blue
            "diminishing hotspot": "orange",
            "diminishing coldspot": "#1E90FF",  # Dodger blue
            "historical hotspot": "#8B4513",  # Saddle brown
            "historical coldspot": "#00008B",  # Dark blue
            "oscilating hotspot": "#FF8C00",  # Dark orange
            "oscilating coldspot": "#0000CD",  # Medium blue
        }

        # Calculate bounds and dimensions
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        width = bounds[2] - bounds[0]  # longitude span
        height = bounds[3] - bounds[1]  # latitude span

        # Calculate optimal zoom level based on the larger dimension
        max_span = max(width, height)

        # New zoom calculation with a more conservative base value
        # and adjusted logarithmic scale for a more zoomed-out view
        zoom = max(4, min(12, -np.log2(max_span) + 9))

        # Additional adjustment for very small or very large areas
        if max_span < 0.01:  # Very small area
            zoom = min(12, zoom + 1)
        elif max_span > 100:  # Very large area
            zoom = max(4, zoom - 1)

        if lib == "plotly":
            # Create the choropleth map
            fig = px.choropleth_mapbox(
                gdf,
                geojson=gdf.geometry.__geo_interface__,
                locations=gdf.index,
                color="classification",
                color_discrete_map=hotspot_colors,
                opacity=0.4,
                hover_data={
                    region_id_field: True,
                    "classification": True,
                    "tau": ":.4f",
                    "p_value": ":.4f",
                },
                custom_data=[gdf["hover_text"]],
                mapbox_style="carto-positron",
                center={
                    "lat": gdf.geometry.centroid.y.mean(),
                    "lon": gdf.geometry.centroid.x.mean(),
                },
            )

            # Update layout with new dynamic zoom
            fig.update_layout(
                title={
                    "text": title,
                    "y": 0.98,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                    "font": {"size": 20},
                },
                margin={"r": 0, "l": 0, "t": 50, "b": 0},
                mapbox={
                    "zoom": zoom,
                    "center": {
                        "lat": gdf.geometry.centroid.y.mean(),
                        "lon": gdf.geometry.centroid.x.mean(),
                    },
                },
                legend={
                    "title": "Classification",
                    "yanchor": "top",
                    "y": 0.99,
                    "xanchor": "left",
                    "x": 1.01,
                    "bgcolor": "rgba(255, 255, 255, 0.8)",
                },
            )

            # Update hover template
            fig.update_traces(
                hovertemplate="%{customdata[0]}<extra></extra>",
                marker_line_width=0.5,
                marker_line_color="#E0E0E0",
            )

            return fig
        elif lib == "folium":
            map_center_coords = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
            # Ensure zoom is an integer for Folium
            folium_zoom = int(zoom)

            folium_map = folium.Map(location=map_center_coords, zoom_start=folium_zoom, tiles="CartoDB positron")

            present_classifications = gdf['classification'].unique()
            
            # Sort items for the legend and layer addition for consistency
            sorted_legend_items_folium = sorted(
                [(label, hotspot_colors[label]) for label in hotspot_colors if label in present_classifications],
                key=lambda item: item[0]
            )

            for classification_name, color_val in sorted_legend_items_folium:
                subset_gdf = gdf[gdf['classification'] == classification_name]
                if not subset_gdf.empty:
                    folium.GeoJson(
                        subset_gdf.__geo_interface__,
                        name=classification_name,
                        style_function=lambda feature, color=color_val: {
                            'fillColor': color,
                            'color': '#E0E0E0',      # Border color similar to Plotly
                            'weight': 0.5,          # Border weight similar to Plotly
                            'fillOpacity': 0.7,
                        },
                        tooltip=folium.features.GeoJsonTooltip(
                            fields=['hover_text'], # Uses the pre-formatted hover_text column
                            aliases=[''],
                            labels=False, # Do not show "hover_text:" as a label
                            sticky=False,
                            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                        ),
                        highlight_function=lambda x: {'weight':2, 'fillOpacity':0.85, 'color': 'black'},
                    ).add_to(folium_map)

            # Create and add HTML legend for Folium
            legend_html = f"""
            <div style="position: fixed;
                        bottom: 20px; left: 20px; width: auto; min-width:180px; max-width:260px;
                        max-height: 300px; border:1px solid grey; z-index:9999; font-size:10px;
                        background-color:rgba(255,255,255,0.85); opacity:0.9; overflow-y:auto; 
                        padding:8px; border-radius:5px; font-family: Arial, sans-serif;">
              <div style="text-align: center; margin-bottom: 5px;"><b>{title}</b></div>
              <div style="text-align: center; margin-bottom: 5px;"><small><i>Classification</i></small></div>
            """
            for classification_name, color_val in sorted_legend_items_folium:
                legend_html += (
                    '<div style="margin-bottom: 3px;">'
                    f'<i style="background:{color_val}; opacity:0.7; border: 1px solid #ccc; display: inline-block; width: 12px; height: 12px; margin-right: 5px; vertical-align: middle;"></i>'
                    f'<span style="vertical-align: middle;">{classification_name}</span>'
                    '</div>'
                )
            legend_html += "</div>"
            folium_map.get_root().html.add_child(folium.Element(legend_html))

            folium.LayerControl(collapsed=True).add_to(folium_map)
            return folium_map
        else:
            raise ValueError("Invalid 'lib' argument. Supported values are 'plotly' or 'folium'.")

    @staticmethod
    def open_visualization_tool(ehsa_results_df=None, auto_open=True):
        """
        Open the EHSA interactive visualization tool in the browser.
        
        Parameters:
        -----------
        ehsa_results_df : pandas.DataFrame, optional
            DataFrame containing EHSA results with geometry column.
            If provided, will save it as a temporary CSV for use in the tool.
        auto_open : bool, default True
            Whether to automatically open the browser. If False, returns the file path.
            
        Returns:
        --------
        str
            Path to the opened HTML file
            
        Example:
        --------
        >>> from pyehsa.ehsa_plotting import EhsaPlotting
        >>> EhsaPlotting.open_visualization_tool()
        
        >>> # Or with data
        >>> EhsaPlotting.open_visualization_tool(ehsa_results_df)
        """
        # Get the path to the HTML file in the package
        current_dir = Path(__file__).parent
        html_source = current_dir / "visualization" / "ehsa_visualization.html"
        
        if not html_source.exists():
            raise FileNotFoundError(
                f"Visualization HTML file not found at {html_source}. "
                "Please ensure the file exists in the visualization directory."
            )
        
        # Create a temporary directory for the visualization
        temp_dir = tempfile.mkdtemp(prefix="ehsa_viz_")
        temp_html = Path(temp_dir) / "ehsa_visualization.html"
        
        # Copy the HTML file to the temporary location
        shutil.copy2(html_source, temp_html)
        
        # If EHSA results are provided, save them as CSV in the same directory
        if ehsa_results_df is not None:
            temp_csv = Path(temp_dir) / "ehsa_results_temp.csv"
            try:
                # Ensure geometry and complex data are in the correct format for the visualization
                df_to_save = ehsa_results_df.copy()
                if 'geometry' in df_to_save.columns:
                    # Convert geometry to GeoJSON-like string if it's not already
                    if hasattr(df_to_save['geometry'].iloc[0], '__geo_interface__'):
                        df_to_save['geometry'] = df_to_save['geometry'].apply(
                            lambda geom: str(geom.__geo_interface__) if geom else None
                        )
                
                # Convert location_data to proper JSON format for easier parsing in HTML
                if 'location_data' in df_to_save.columns:
                    df_to_save['location_data'] = df_to_save['location_data'].apply(
                        lambda x: json.dumps(x) if x is not None else None
                    )
                
                # Convert other complex columns to JSON format as well  
                complex_columns = ['classification_details', 'mann_kendall_details', 'spatial_context_summary']
                for col in complex_columns:
                    if col in df_to_save.columns:
                        df_to_save[col] = df_to_save[col].apply(
                            lambda x: json.dumps(x) if x is not None else None
                        )
                
                # Sanitize all values to prevent CSV formula injection attacks
                df_to_save = EhsaPlotting._sanitize_dataframe_for_csv(df_to_save)
                
                df_to_save.to_csv(temp_csv, index=False)
                print(f"EHSA results saved to: {temp_csv}")
                print("You can load this file in the visualization tool that will open.")
            except Exception as e:
                print(f"Warning: Could not save EHSA results to CSV: {e}")
        
        # Convert to file:// URL for browser opening
        file_url = temp_html.as_uri()
        
        if auto_open:
            try:
                # Try to open in the default browser
                webbrowser.open(file_url)
                print(f"Opening EHSA Visualization Tool in your default browser...")
                print(f"File location: {temp_html}")
                if ehsa_results_df is not None:
                    print("\nTo use your data:")
                    print("1. Click 'Choose File' in the visualization tool")
                    print(f"2. Navigate to: {temp_dir}")
                    print("3. Select 'ehsa_results_temp.csv'")
            except Exception as e:
                print(f"Could not automatically open browser: {e}")
                print(f"Please manually open this file in your browser: {file_url}")
        else:
            print(f"Visualization tool prepared at: {temp_html}")
            print(f"Open this URL in your browser: {file_url}")
        
        return str(temp_html)
    
    @staticmethod
    def save_ehsa_visualization_tool(file_path, ehsa_results_df=None, include_data=True):
        """
        Save the EHSA interactive visualization tool to a specified HTML file.
        Similar to folium.save() or plotly.write_html() - saves without opening browser.
        
        Parameters:
        -----------
        file_path : str or Path
            Path where to save the HTML file (e.g., '/path/to/my_ehsa_viz.html')
        ehsa_results_df : pandas.DataFrame, optional
            DataFrame containing EHSA results with geometry column.
            If provided, will embed the data directly in the HTML file.
        include_data : bool, default True
            Whether to include the EHSA data directly in the HTML file.
            If True, the data will be embedded and no separate CSV is needed.
            If False, the HTML will be saved without data (user must upload CSV manually).
            
        Returns:
        --------
        str
            Absolute path to the saved HTML file
            
        Example:
        --------
        >>> from pyehsa.ehsa_plotting import EhsaPlotting
        >>> # Save with embedded data
        >>> EhsaPlotting.save_ehsa_visualization_tool(
        ...     'my_ehsa_visualization.html', 
        ...     ehsa_results_df=results
        ... )
        
        >>> # Save empty tool for manual data loading
        >>> EhsaPlotting.save_ehsa_visualization_tool(
        ...     'ehsa_tool.html', 
        ...     include_data=False
        ... )
        """
        # Convert file_path to Path object and get absolute path
        save_path = Path(file_path).resolve()
        
        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving EHSA Visualization Tool to: {save_path}")
        
        # Get the path to the HTML template
        current_dir = Path(__file__).parent
        html_source = current_dir / "visualization" / "ehsa_visualization.html"
        
        if not html_source.exists():
            raise FileNotFoundError(
                f"Visualization HTML template not found at {html_source}. "
                "Please ensure the file exists in the visualization directory."
            )
        
        # Read the HTML template
        with open(html_source, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # If data is provided and should be included, embed it in the HTML
        if ehsa_results_df is not None and include_data:
            try:
                # Process the data for embedding
                df_processed = ehsa_results_df.copy()
                
                # Convert geometry to GeoJSON format
                if 'geometry' in df_processed.columns:
                    if hasattr(df_processed['geometry'].iloc[0], '__geo_interface__'):
                        df_processed['geometry'] = df_processed['geometry'].apply(
                            lambda geom: geom.__geo_interface__ if geom else None
                        )
                
                # Convert complex columns to JSON format for JavaScript consumption
                complex_columns = ['location_data', 'classification_details', 'mann_kendall_details', 'spatial_context_summary']
                for col in complex_columns:
                    if col in df_processed.columns:
                        df_processed[col] = df_processed[col].apply(
                            lambda x: x if x is None else (x if isinstance(x, (dict, list)) else x)
                        )
                
                # Convert DataFrame to JSON for embedding
                data_json = df_processed.to_json(orient='records')
                
                # Escape </script> to prevent script breakout XSS attacks
                # Replace with <\/script> which is safe in JavaScript strings
                data_json_safe = data_json.replace('</script>', r'<\/script>')
                
                # Create JavaScript code to automatically load the data
                auto_load_script = f"""
        <script>
            // Auto-load embedded EHSA data
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('Loading embedded EHSA data...');
                
                // Embedded data (script breakout protected)
                const embeddedData = {data_json_safe};
                
                // Convert to GeoJSON format
                const geojson = {{
                    type: 'FeatureCollection',
                    features: embeddedData.map(row => ({{
                        type: 'Feature',
                        geometry: row.geometry,
                        properties: Object.fromEntries(
                            Object.entries(row).filter(([key, value]) => {{
                                // Filter out geometry and dangerous prototype pollution keys
                                const dangerousKeys = ['__proto__', 'constructor', 'prototype'];
                                return key !== 'geometry' && !dangerousKeys.includes(key);
                            }})
                        )
                    }}))
                }};
                
                console.log('Embedded data converted to GeoJSON:', geojson);
                
                // Load the data into the map
                loadGeoJSON(geojson);
                
                // Hide the upload section since data is already loaded
                const uploadSection = document.querySelector('.row .col-12 .card');
                if (uploadSection) {{
                    uploadSection.style.display = 'none';
                }}
                
                // Add a note about embedded data
                const title = document.querySelector('h1');
                if (title) {{
                    const note = document.createElement('p');
                    note.className = 'text-center text-muted';
                    note.innerHTML = '<small>📊 This visualization includes embedded EHSA results data</small>';
                    title.parentNode.insertBefore(note, title.nextSibling);
                }}
            }});
        </script>
        """
                
                # Insert the auto-load script before the closing body tag
                html_content = html_content.replace('</body>', auto_load_script + '\n</body>')
                
                print(f"✅ EHSA data embedded in HTML file ({len(ehsa_results_df)} regions)")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not embed data in HTML: {e}")
                print("Saving HTML without embedded data. You can load data manually.")
        
        elif ehsa_results_df is not None and not include_data:
            print("ℹ️  Data provided but include_data=False. Saving HTML without embedded data.")
        
        # Write the HTML file
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ EHSA Visualization Tool saved successfully!")
            print(f"📁 File location: {save_path}")
            print(f"📊 File size: {save_path.stat().st_size / 1024:.1f} KB")
            
            if ehsa_results_df is None or not include_data:
                print("\n💡 Usage instructions:")
                print("1. Open the HTML file in your web browser")
                print("2. Use the 'Upload EHSA Results (CSV)' section to load your data")
                print("3. Export your EHSA results using: EhsaPlotting.open_visualization_tool(results, auto_open=False)")
            else:
                print("\n💡 Usage instructions:")
                print("1. Open the HTML file in your web browser")
                print("2. Data is already loaded and ready for exploration!")
            
            return str(save_path)
            
        except Exception as e:
            raise IOError(f"Failed to save HTML file to {save_path}: {e}")
    
    @staticmethod 
    def launch_ehsa_viewer():
        """
        Convenience method to quickly launch the EHSA visualization tool.
        This is a simplified alias for open_visualization_tool().
        
        Example:
        --------
        >>> from pyehsa.ehsa_plotting import EhsaPlotting
        >>> EhsaPlotting.launch_ehsa_viewer()
        """
        return EhsaPlotting.open_visualization_tool()
