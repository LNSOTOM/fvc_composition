import leafmap.foliumap as leafmap
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import folium

def plot_blocks_folds(coordinates, block_labels, fold_assignments, crs="EPSG:7854"):
    # Create a GeoDataFrame for the coordinates
    points = [Point(x, y) for x, y in coordinates]
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    
    # Assign block labels to the GeoDataFrame
    gdf['block'] = block_labels
    
    # Reproject to EPSG:4326 for compatibility with web maps
    gdf = gdf.to_crs("EPSG:4326")

    # Center the map around the mean of the points
    center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
    m = leafmap.Map(center=center, zoom=8, google_map="TERRAIN")

    # Add a satellite basemap as an option
    m.add_basemap('SATELLITE')

    # Define a color palette for blocks and points
    block_colors = ["#03AED2", "#399918", "#4F1787", "#921A40", "#F7C566"]
    train_color = "#EB3678"  # Red for Train
    val_color = "#8DECB4"  # Green for Val
    test_color = "#615EFC"  # Purple for Test

    legend_labels = []
    color_list = []

    # Plot blocks and data splits
    available_blocks = np.unique(block_labels)

    for block in available_blocks:
        block_gdf = gdf[gdf['block'] == block]
        block_color = block_colors[block % len(block_colors)]
        
        # Create a layer group for the block
        block_layer = folium.FeatureGroup(name=f'Block {block}', show=True)

        # Add block points with custom colors using folium.CircleMarker
        for _, row in block_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=6,
                color=block_color,
                fill=True,
                fill_color=block_color,
                fill_opacity=0.7,
                popup=f'Block {block}',
            ).add_to(block_layer)

        block_layer.add_to(m)
        legend_labels.append(f'Block {block}')
        color_list.append(block_color)

        # Retrieve train, validation, and test indices for the current block
        indices_dict = fold_assignments.get(block, {})

        train_indices = indices_dict.get('train_indices', [])
        val_indices = indices_dict.get('val_indices', [])
        test_indices = indices_dict.get('test_indices', [])

        # Create layer groups for Train, Val, and Test
        train_layer = folium.FeatureGroup(name=f'Fold {block} - Train', show=False)
        val_layer = folium.FeatureGroup(name=f'Fold {block} - Val', show=False)
        test_layer = folium.FeatureGroup(name=f'Fold {block} - Test', show=False)

        # Plot Train points
        for idx in train_indices:
            if idx < len(gdf):
                row = gdf.iloc[idx]
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=train_color,
                    fill=True,
                    fill_color=train_color,
                    fill_opacity=0.7,
                    popup=f'Block {block} - Train',
                ).add_to(train_layer)

        # Plot Val points
        for idx in val_indices:
            if idx < len(gdf):
                row = gdf.iloc[idx]
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=val_color,
                    fill=True,
                    fill_color=val_color,
                    fill_opacity=0.7,
                    popup=f'Block {block} - Val',
                ).add_to(val_layer)

        # Plot Test points
        for idx in test_indices:
            if idx < len(gdf):
                row = gdf.iloc[idx]
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=6,
                    color=test_color,
                    fill=True,
                    fill_color=test_color,
                    fill_opacity=0.7,
                    popup=f'Block {block} - Test',
                ).add_to(test_layer)

        # Add layers to the map
        train_layer.add_to(m)
        val_layer.add_to(m)
        test_layer.add_to(m)

    # Create a custom legend for blocks and data splits
    legend_items = "".join(
        [f'<i class="fa fa-square" style="color:{color_list[i]};"></i> {legend_labels[i]}<br>' for i in range(len(legend_labels))]
    )
    legend_items += f'<i class="fa fa-circle" style="color:{train_color};"></i> Train Set<br>'
    legend_items += f'<i class="fa fa-circle" style="color:{val_color};"></i> Validation Set<br>'
    legend_items += f'<i class="fa fa-circle" style="color:{test_color};"></i> Test Set<br>'

    legend_html = f'''
    <div id="legend" style="position: fixed; 
                bottom: 0; left: 20px; width: 250px; height: auto; margin-left: 50px;
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
        <div id="legend-content" style="padding: 10px;">
            <b>Legend</b><br>
            {legend_items}
        </div>
        <button onclick="toggleLegend()" style="width: 100%; margin-top: 0px;">Hide Legend</button>
    </div>
    <script>
        function toggleLegend() {{
            var content = document.getElementById("legend-content");
            var button = document.querySelector("#legend button");

            if (content.style.display === "none") {{
                content.style.display = "block";
                button.innerHTML = "Hide Legend";
            }} else {{
                content.style.display = "none";
                button.innerHTML = "Show Legend";
            }}
        }}
    </script>
    '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add LayerControl to allow toggling of the layers
    folium.LayerControl().add_to(m)

    # Save the map to an HTML file
    m.to_html('block_folds_map.html')
    print("Map saved to block_folds_map.html")
