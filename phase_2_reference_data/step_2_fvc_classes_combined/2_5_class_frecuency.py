
# %%
# site-specific models
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, scale_alpha_manual, position_dodge, guides, guide_legend

# Sample DataFrame structured for bar plot and faceting by category
data = {
    'class': ['BE', 'NPV', 'PV', 'SI', 'WI'] * 6,  # 5 classes, repeated twice for each density (2 x 3 = 6)
    'frequency': [
        # LOW category frequencies
        0.39558546, 0.35216222, 0.25225232, 0.0, 0.0,  # LOW - Before
        0.36854068, 0.36701858, 0.26444074, 0.0, 0.0,  # LOW - After
        # MEDIUM category frequencies
        0.41197585, 0.03871381, 0.34499355, 0.20431679, 0.0,  # MEDIUM - Before
        0.35891044, 0.0410812, 0.37744875, 0.22255962, 0.0,  # MEDIUM - After
        # DENSE category frequencies
        0.24742084, 0.01648553, 0.43895739, 0.13097304, 0.1661632,  # DENSE - Before
        0.21666717, 0.01612868, 0.45556296, 0.13681792, 0.17482327  # DENSE - After
    ],
    'density': (['LOW'] * 5 + ['LOW'] * 5 +  # LOW - Before and After
                ['MEDIUM'] * 5 + ['MEDIUM'] * 5 +  # MEDIUM - Before and After
                ['DENSE'] * 5 + ['DENSE'] * 5),  # DENSE - Before and After
    'type': ['Before'] * 5 + ['After'] * 5 +  # LOW - Before and After
            ['Before'] * 5 + ['After'] * 5 +  # MEDIUM - Before and After
            ['Before'] * 5 + ['After'] * 5   # DENSE - Before and After
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Set order for density and type to ensure correct plotting sequence
df['density'] = pd.Categorical(df['density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
df['type'] = pd.Categorical(df['type'], categories=['Before', 'After'], ordered=True)

# Custom colors for each class
class_colors = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}

# Define the dodge position with a specified width to create space between Before and After within each class
dodge_position = position_dodge(width=0.75)  # Increase this width to create more space between Before and After

# Create the plot
plot = (ggplot(df, aes(x='class', y='frequency', fill='class'))
        + geom_col(aes(alpha='type', linetype='type'), position=dodge_position, color='black', width=0.7)  # Use alpha based on type for transparency in bars only
        # Add "Before" labels with different nudge values
        + geom_text(df[df['type'] == 'Before'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='right', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for Before text
        # Add "After" labels with different nudge values and rotated text
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='left', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for After text
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + scale_alpha_manual(values={'Before': 1.0, 'After': 0.5})  # Full opacity for 'Before', 50% transparency for 'After'
        + guides(linetype=guide_legend(override_aes={'color': 'black', 'fill': 'none'}),  # Outline only for Before/After, with filled class colors
                 fill=guide_legend(override_aes={'color': None}))  # Keep the fill for the class colors
        + theme(axis_text_x=element_text(size=12, rotation=0, ha='center'),  # Set font size for x-axis labels
                axis_text_y=element_text(size=12),
                axis_title_x=element_text(size=14),  # Set font size for x-axis title
                axis_title_y=element_text(size=14),  # Set font size for y-axis title
                strip_text=element_text(size=12),  # Set font size for the facet labels (LOW, MEDIUM, DENSE)
                legend_title=element_text(size=12),
                legend_text=element_text(size=12),  # Set font size for legend text
                figure_size=(12, 8))  # Set figure size (width, height)
       )

# Save the plot as a high-resolution PNG file
plot.save("class_frequency.png", dpi=300, width=12, height=8)

print(plot)


#%%
##version 2
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, scale_alpha_manual, position_dodge, guides, guide_legend, element_rect, element_line, theme_bw

# Sample DataFrame structured for bar plot and faceting by category
data = {
    'class': ['BE', 'NPV', 'PV', 'SI', 'WI'] * 6,  # 5 classes, repeated twice for each density (2 x 3 = 6)
    'frequency': [
        # LOW category frequencies
        0.39558546, 0.35216222, 0.25225232, 0.0, 0.0,  # LOW - Before
        0.36854068, 0.36701858, 0.26444074, 0.0, 0.0,  # LOW - After
        # MEDIUM category frequencies
        0.41197585, 0.03871381, 0.34499355, 0.20431679, 0.0,  # MEDIUM - Before
        0.35891044, 0.0410812, 0.37744875, 0.22255962, 0.0,  # MEDIUM - After
        # DENSE category frequencies
        0.24742084, 0.01648553, 0.43895739, 0.13097304, 0.1661632,  # DENSE - Before
        0.21666717, 0.01612868, 0.45556296, 0.13681792, 0.17482327  # DENSE - After
    ],
    'density': (['LOW'] * 5 + ['LOW'] * 5 +  # LOW - Before and After
                ['MEDIUM'] * 5 + ['MEDIUM'] * 5 +  # MEDIUM - Before and After
                ['DENSE'] * 5 + ['DENSE'] * 5),  # DENSE - Before and After
    'type': ['Before'] * 5 + ['After'] * 5 +  # LOW - Before and After
            ['Before'] * 5 + ['After'] * 5 +  # MEDIUM - Before and After
            ['Before'] * 5 + ['After'] * 5   # DENSE - Before and After
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Set order for density and type to ensure correct plotting sequence
df['density'] = pd.Categorical(df['density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
df['type'] = pd.Categorical(df['type'], categories=['Before', 'After'], ordered=True)

# Custom colors for each class
class_colors = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}

# Define the dodge position with a specified width to create space between Before and After within each class
dodge_position = position_dodge(width=0.75)  # Increase this width to create more space between Before and After

# Create the plot
plot = (ggplot(df, aes(x='class', y='frequency', fill='class'))
        + geom_col(aes(alpha='type', linetype='type'), position=dodge_position, color='black', width=0.7)  # Use alpha based on type for transparency in bars only
        # Add "Before" labels with different nudge values
        + geom_text(df[df['type'] == 'Before'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='right', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for Before text
        # Add "After" labels with different nudge values and rotated text
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='left', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for After text
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + scale_alpha_manual(values={'Before': 1.0, 'After': 0.5})  # Full opacity for 'Before', 50% transparency for 'After'
        + guides(linetype=guide_legend(override_aes={'color': 'black', 'fill': 'none'}),  # Outline only for Before/After, with filled class colors
                 fill=guide_legend(override_aes={'color': None}))  # Keep the fill for the class colors
        + theme_bw()  # Apply the black-and-white theme
        + theme(
            panel_background=element_rect(fill='white', color='black'),  # White panel background with a black border
            plot_background=element_rect(fill='white'),  # White plot background
            # panel_grid_major=element_line(color='lightgrey'),  # Dotted grid lines for major grids: linetype='dashed'
            # panel_grid_minor=element_line(color='lightgrey'),  # Dashed grid lines for minor grids
            strip_background=element_rect(fill='#FFF6E9', color='black'),  # Light blue background for facet grid labels: fill='black', color='white'
            strip_text=element_text(size=12, color='black'),  # Set font size for the facet labels (LOW, MEDIUM, DENSE): color='white'
            axis_text_x=element_text(size=12, rotation=0, ha='center'),  # Set font size for x-axis labels
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=14),  # Set font size for x-axis title
            axis_title_y=element_text(size=14),  # Set font size for y-axis title           
            legend_title=element_text(size=12),
            legend_text=element_text(size=12),  # Set font size for legend text
            figure_size=(12, 8))  # Set figure size (width, height)
       )


# Save the plot as a high-resolution PNG file
plot.save("class_frequency.png", dpi=300, width=12, height=8)

print(plot)


# %%
### Across sites
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, scale_alpha_manual, position_dodge, guides, guide_legend,  element_rect, theme_bw

# Sample DataFrame for the "Sites" dataset
data_sites = {
    'class': ['BE', 'NPV', 'PV', 'SI', 'WI'] * 2,  # 5 classes, repeated for Before and After
    'frequency': [
        0.3344477, 0.09851538, 0.36653346, 0.12590219, 0.07460127,  # Original frequencies ("Before")
        0.29609293, 0.10352324, 0.38741802, 0.13280446, 0.08016135   # Subsampled frequencies ("After")
    ],
    'density': ['SITES'] * 10,  # All rows labeled as "Sites"
    'type': ['Before'] * 5 + ['After'] * 5  # "Before" for the first set, "After" for the second set
}

# Convert the data into a DataFrame
df_sites = pd.DataFrame(data_sites)

# Set order for type to ensure correct plotting sequence
df_sites['type'] = pd.Categorical(df_sites['type'], categories=['Before', 'After'], ordered=True)

# Custom colors for each class
class_colors = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}

# Define the dodge position with a specified width to create space between Before and After within each class
dodge_position = position_dodge(width=0.52)

# Create the plot
sites_plot = (ggplot(df_sites, aes(x='class', y='frequency', fill='class'))
        + geom_col(aes(alpha='type', linetype='type'), position=dodge_position, color='black', width=0.5)
        # Add "Before" labels with different nudge values
        + geom_text(df_sites[df_sites['type'] == 'Before'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='center', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=-0.2)  # Specific nudge for Before text
        # Add "After" labels with different nudge values and rotated text
        + geom_text(df_sites[df_sites['type'] == 'After'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='left', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for After text
        + facet_grid('~density')  # Facet by density (only "Sites" here)
        + labs(x='Class', y='Frequency', title='')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + scale_alpha_manual(values={'Before': 1.0, 'After': 0.5})  # Full opacity for 'Before', 50% transparency for 'After'
        + guides(linetype=guide_legend(override_aes={'color': 'black', 'fill': 'none'}),  # Outline only for Before/After
                 fill=guide_legend(override_aes={'color': None}))  # Keep the fill for the class colors
        + theme_bw()  # Apply the black-and-white theme
        + theme(
            panel_background=element_rect(fill='white', color='black'),  # White panel background with a black border
            plot_background=element_rect(fill='white'),  # White plot background
            # panel_grid_major=element_line(color='lightgrey'),  # Dotted grid lines for major grids: linetype='dashed'
            # panel_grid_minor=element_line(color='lightgrey'),  # Dashed grid lines for minor grids
            strip_background=element_rect(fill='#FFF6E9', color='black'),  # Light blue background for facet grid labels: fill='black', color='white'
            strip_text=element_text(size=12, color='black'),  # Set font size for the facet labels (LOW, MEDIUM, DENSE): color='white'
            axis_text_x=element_text(size=12, rotation=0, ha='center'),  # Set font size for x-axis labels
            axis_text_y=element_text(size=12),
            axis_title_x=element_text(size=14),  # Set font size for x-axis title
            axis_title_y=element_text(size=14),  # Set font size for y-axis title
            legend_title=element_text(size=12),
            legend_text=element_text(size=12),  # Set font size for legend text
            figure_size=(12, 8))  # Set figure size (width, height)
       )

# Save the plot as a high-resolution PNG file
sites_plot.save("class_frequency_sites.png", dpi=300, width=12, height=8)


# Display the plot inline
print(sites_plot)
# %%
