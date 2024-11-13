
#%%
import matplotlib.pyplot as plt
import numpy as np

# Data from before and after subsampling
class_labels = ['Class 1', 'Class 2', 'Class 3']
original_frequencies = [0.39558546, 0.35216222, 0.25225232]
subsampled_frequencies = [0.36854068, 0.36701858, 0.26444074]

# Set up the figure and axes for two bar plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot for original class frequencies
ax[0].bar(class_labels, original_frequencies, color='blue', alpha=0.7)
ax[0].set_ylim(0, 0.5)
ax[0].set_title('Class Frequencies Before Subsampling')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Class')

# Plot for subsampled class frequencies
ax[1].bar(class_labels, subsampled_frequencies, color='green', alpha=0.7)
ax[1].set_ylim(0, 0.5)
ax[1].set_title('Class Frequencies After Subsampling')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Class')

# Show the plot
plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Apply ggplot style
plt.style.use('ggplot')

# Data from before and after subsampling
class_labels = ['BE', 'NPV', 'PV']  # Updated class labels
original_frequencies = [0.39558546, 0.35216222, 0.25225232]
subsampled_frequencies = [0.36854068, 0.36701858, 0.26444074]
class_colors = ['#dae22f', '#6332ea', '#e346ee']  # Custom colors for each class

# Define the positions for the bars
x = np.arange(len(class_labels))  # Label locations
width = 0.35  # Width of the bars

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot for original class frequencies
bars_before = ax.bar(x - width/2, original_frequencies, width, label='Before Subsampling', color=class_colors)

# Plot for subsampled class frequencies with a diamond pattern
bars_after = ax.bar(x + width/2, subsampled_frequencies, width, label='After Subsampling', color=class_colors, hatch='x')  # Using 'x' for diamond-like pattern

# Add labels on top of each bar
for bar in bars_before:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

for bar in bars_after:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

# Add labels, title, and legend
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.set_ylim(0, 0.5)
ax.set_ylabel('Frequency')
ax.set_xlabel('Class')
ax.set_title('Class Frequencies Before and After Subsampling')
ax.legend()

# Show plot
plt.tight_layout()
plt.show()




# %%
import matplotlib.pyplot as plt
import numpy as np

# Apply ggplot style
plt.style.use('ggplot')

# Data from before and after subsampling
class_labels = ['BE', 'NPV', 'PV']  # Updated class labels
original_frequencies = [0.39558546, 0.35216222, 0.25225232]
subsampled_frequencies = [0.36854068, 0.36701858, 0.26444074]
class_colors = ['#dae22f', '#6332ea', '#e346ee']  # Custom colors for each class

# Set up the figure with multiple subplots for each class under the "LOW" group
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
fig.suptitle('Class Frequencies Before and After Subsampling - Group: LOW', fontsize=16)

# Plot for each class in separate panels
for i, label in enumerate(class_labels):
    axs[i].bar(0, original_frequencies[i], width=0.35, label='Before', color=class_colors[i])
    axs[i].bar(1, subsampled_frequencies[i], width=0.35, label='After', color=class_colors[i], hatch='x')
    
    # Set labels and title for each subplot
    axs[i].set_xticks([0, 1])
    axs[i].set_xticklabels(['Before', 'After'])
    axs[i].set_ylim(0, 0.5)
    axs[i].set_title(f'Class: {label}')
    axs[i].set_ylabel('Frequency')
    
    # Add values on top of each bar
    axs[i].text(0, original_frequencies[i] + 0.01, f'{original_frequencies[i]:.3f}', ha='center', va='bottom')
    axs[i].text(1, subsampled_frequencies[i] + 0.01, f'{subsampled_frequencies[i]:.3f}', ha='center', va='bottom')

# Add a single legend for all subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Apply ggplot style
plt.style.use('ggplot')

# Data for LOW, MEDIUM, and DENSE categories with updated class labels and colors
class_labels = ['BE', 'NPV', 'PV', 'SI', 'WI']  # Updated class labels
low_original_frequencies = [0.39558546, 0.35216222, 0.25225232, 0.0, 0.0]
low_subsampled_frequencies = [0.36854068, 0.36701858, 0.26444074, 0.0, 0.0]
medium_original_frequencies = [0.41197585, 0.03871381, 0.34499355, 0.20431679, 0.0]
medium_subsampled_frequencies = [0.35891044, 0.0410812, 0.37744875, 0.22255962, 0.0]
dense_original_frequencies = [0.24742084, 0.01648553, 0.43895739, 0.13097304, 0.1661632]
dense_subsampled_frequencies = [0.21666717, 0.01612868, 0.45556296, 0.13681792, 0.17482327]
class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Colors for each class

# Set up the figure with 3 rows for LOW, MEDIUM, and DENSE groups
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
fig.suptitle('Class Frequencies Before and After Subsampling - Group Faceting by LOW, MEDIUM, and DENSE', fontsize=16)

# Plot for LOW group
x_low = np.arange(len(low_original_frequencies))  # Position for LOW classes
width = 0.35  # Width of the bars

axs[0].bar(x_low - width/2, low_original_frequencies, width, color=class_colors, label='Before Subsampling')
axs[0].bar(x_low + width/2, low_subsampled_frequencies, width, color=class_colors, hatch='x', label='After Subsampling')

# Adding text labels above bars for LOW
for i, (orig, subs) in enumerate(zip(low_original_frequencies, low_subsampled_frequencies)):
    axs[0].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
    axs[0].text(i + width/2, subs + 0.01, f'{subs:.3f}', ha='center', va='bottom')

axs[0].set_title('LOW')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Plot for MEDIUM group
x_medium = np.arange(len(medium_original_frequencies))  # Position for MEDIUM classes

axs[1].bar(x_medium - width/2, medium_original_frequencies, width, color=class_colors, label='Before Subsampling')
axs[1].bar(x_medium + width/2, medium_subsampled_frequencies, width, color=class_colors, hatch='x', label='After Subsampling')

# Adding text labels above bars for MEDIUM
for i, (orig, subs) in enumerate(zip(medium_original_frequencies, medium_subsampled_frequencies)):
    axs[1].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
    axs[1].text(i + width/2, subs + 0.01, f'{subs:.3f}', ha='center', va='bottom')

axs[1].set_title('MEDIUM')
axs[1].set_ylabel('Frequency')

# Plot for DENSE group
x_dense = np.arange(len(dense_original_frequencies))  # Position for DENSE classes

axs[2].bar(x_dense - width/2, dense_original_frequencies, width, color=class_colors, label='Before Subsampling')
axs[2].bar(x_dense + width/2, dense_subsampled_frequencies, width, color=class_colors, hatch='x', label='After Subsampling')

# Adding text labels above bars for DENSE
for i, (orig, subs) in enumerate(zip(dense_original_frequencies, dense_subsampled_frequencies)):
    axs[2].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
    axs[2].text(i + width/2, subs + 0.01, f'{subs:.3f}', ha='center', va='bottom')

axs[2].set_title('DENSE')
axs[2].set_ylabel('Frequency')
axs[2].set_xlabel('Class')
axs[2].set_xticks(x_dense)
axs[2].set_xticklabels(class_labels)

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

# Set up the FacetGrid for bar plot faceted by 'density'
g = sns.catplot(
    data=df, x='class', y='frequency', hue='type', col='density',
    kind='bar', height=5, aspect=0.8, palette=['#63B2FF', '#FF6363']
)

# Apply hatch pattern manually to 'After' bars
for ax in g.axes.flat:
    # Iterate over all bars
    for bar, type_ in zip(ax.patches, df['type']):
        # Apply hatch only if the bar corresponds to 'After'
        if type_ == 'After':
            bar.set_hatch('x')

# Customize the plot to match ggplot-style theme
g.set_axis_labels("Class", "Frequency")
g.set_titles("{col_name}")  # Simplify titles to show only 'density' level
g.add_legend(title="Type")

# Rotate x-axis labels for readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_fontsize(10)

# Show plot
plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data setup with custom colors for each class
class_labels = ['BE', 'NPV', 'PV', 'SI', 'WI']
class_colors = ['#dae22f', '#6332ea', '#e346ee', '#6da4d4', '#68e8d3']  # Custom colors for each class

# Sample DataFrame structured for bar plot and faceting by category
data = {
    'class': class_labels * 6,  # 5 classes, repeated twice for each density (2 x 3 = 6)
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

# Set up the FacetGrid for bar plot faceted by 'density'
g = sns.catplot(
    data=df, x='class', y='frequency', hue='type', col='density',
    kind='bar', height=5, aspect=0.8, palette=['#636363', '#636363']  # Temporary palette
)

# Apply individual class colors and hatch patterns for 'After' bars
for ax in g.axes.flat:
    for bar, (class_, type_) in zip(ax.patches, zip(df['class'], df['type'])):
        # Set the color based on class
        color = class_colors[class_labels.index(class_)]
        bar.set_facecolor(color)
        
        # Apply hatch pattern only for 'After' bars
        if type_ == 'After':
            bar.set_hatch('x')

# Customize the plot with labels and legend
g.set_axis_labels("Class", "Frequency")
g.set_titles("{col_name}")  # Show density level as the title
g.add_legend(title="Type")

# Rotate x-axis labels for readability
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(10)

# Show plot
plt.tight_layout()
plt.show()

# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, theme, element_text, scale_fill_manual, position_dodge

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

# Define the dodge position with a specified width to increase space between Before and After
dodge_position = position_dodge(width=0.9)

# Create the plot
plot = (ggplot(df, aes(x='class', y='frequency', fill='class'))
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.9)  # Adjusted width for more spacing
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='Class Frequencies Before and After Subsampling by Density')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + theme(axis_text_x=element_text(rotation=0, ha='center'), legend_title=element_text(size=10))
       )

# Display the plot
print(plot)



# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, theme, element_text, scale_fill_manual, position_dodge

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
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.7)  # Adjusted width of bars to maintain separation
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='Class Frequencies Before and After Subsampling by Density')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + theme(axis_text_x=element_text(rotation=0, ha='center'), legend_title=element_text(size=10))
       )

# Display the plot
print(plot)


# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, position_dodge

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
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.7)  # Adjusted width of bars to maintain separation
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  # Only add text labels to 'After' bars
                    position=dodge_position, angle=35, ha='left', va='bottom', format_string='{:.2f}', size=8, nudge_x=0.01, nudge_y=0.00)  # Adjust label position
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='Class Frequencies Before and After Subsampling by Density')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + theme(axis_text_x=element_text(rotation=0, ha='center'), legend_title=element_text(size=10))
       )

# Display the plot
print(plot)


# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, position_dodge

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
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.7)  # Adjusted width of bars to maintain separation
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  # Only add text labels to 'After' bars
                    position=dodge_position, angle=45, ha='left', va='bottom', format_string='{:.3f}', 
                    size=9, nudge_y=0.00, nudge_x=0.1)  # Rotate labels by 45 degrees and nudge to the right
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='Class Frequencies Before and After Subsampling by Density')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + theme(axis_text_x=element_text(size=12, rotation=0, ha='center'), legend_title=element_text(size=14),
                figure_size=(12, 8))  # Set figure size (width, height)
       )

# Display the plot
# print(plot)

# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, position_dodge

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
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.7)  # Adjusted width of bars to maintain separation
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  # Only add text labels to 'After' bars
                    position=dodge_position, angle=45, ha='left', va='bottom', format_string='{:.2f}', 
                    size=9, nudge_y=0.00, nudge_x=0.1)  # Rotate labels by 45 degrees and nudge to the right
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
        + theme(axis_text_x=element_text(size=12, rotation=0, ha='center'),  # Set font size for x-axis labels
                axis_text_y=element_text(size=12),
                axis_title_x=element_text(size=14),  # Set font size for x-axis title
                axis_title_y=element_text(size=14),  # Set font size for y-axis title
                strip_text=element_text(size=12),  # Set font size for the facet labels (LOW, MEDIUM, DENSE)
                legend_title=element_text(size=12),
                legend_text=element_text(size=12),  # Set font size for legend text
                # plot_title=element_text(size=16, weight='bold'),  # Set font size for the plot title 
                figure_size=(10, 7))  # Set figure size (width, height)
       )

# Save the plot as a high-resolution PNG file
plot.save("class_frequencies.png", dpi=300, width=10, height=7)

# %%
import pandas as pd
from plotnine import ggplot, aes, facet_grid, labs, geom_col, geom_text, theme, element_text, scale_fill_manual, position_dodge, guides, guide_legend

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
        + geom_col(aes(linetype='type'), position=dodge_position, color='black', width=0.7)  # Adjusted width of bars to maintain separation
        # Add "Before" labels with different nudge values
        + geom_text(df[df['type'] == 'Before'], aes(label='frequency', y='frequency'),  
                    position=dodge_position, angle=35, ha='right', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Specific nudge for Before
        # Add "After" labels with different nudge values and rotated text
        + geom_text(df[df['type'] == 'After'], aes(label='frequency', y='frequency'),  # Only add text labels to 'After' bars
                    position=dodge_position, angle=35, ha='left', va='bottom', format_string='{:.2f}', 
                    size=10, nudge_y=0.003, nudge_x=0.1)  # Rotate labels by 45 degrees and nudge to the right
        + facet_grid('~density')  # Facet by density with specified order
        + labs(x='Class', y='Frequency', title='')
        + scale_fill_manual(values=class_colors)  # Apply custom colors using scale_fill_manual
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


# Display the plot
# print(plot)
# Save the plot as a high-resolution PNG file
plot.save("class_frequencies.png", dpi=300, width=12, height=8)


# %%
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

# %%
