
#%%
## version 1: F1-Scores
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_point, labs, theme, element_text, scale_fill_manual, scale_color_manual, coord_cartesian, facet_grid, position_dodge, stat_summary

# Prepare data function that works independently for each density
def prepare_data(val_scores, test_scores, classes, val_counts, test_counts, density_label):
    data = pd.DataFrame({
        'Class': classes * 6,
        'F1_Score': sum(val_scores, []) + sum(test_scores, []),
        'Evaluation set': ['VALIDATION'] * 3 * len(classes) + ['TEST'] * 3 * len(classes),
        'Density': [density_label] * 6 * len(classes)
    })
    
    # Force the categorical order here
    data['Evaluation set'] = pd.Categorical(data['Evaluation set'], categories=['VALIDATION', 'TEST'], ordered=True)
    data['Density'] = pd.Categorical(data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
    
    # Calculate means for each class in the density group
    means_data = data.groupby(['Class', 'Evaluation set'], as_index=False)['F1_Score'].mean()
    means_data.rename(columns={'F1_Score': 'Mean_F1_Score'}, inplace=True)
    means_data['Density'] = density_label

    # Prepare counts_data
    counts_data = pd.DataFrame({
        'Class': classes * 2,
        'Counts': [f'V= {val_counts[cls]:.2f}%' for cls in classes] +
                  [f'T= {test_counts[cls]:.2f}%' for cls in classes],
        'y_position': [1.03] * len(classes) + [1.00] * len(classes),
        'Evaluation set': ['VALIDATION'] * len(classes) + ['TEST'] * len(classes),
        'Density': [density_label] * 2 * len(classes)
    })
    counts_data['Density'] = pd.Categorical(counts_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)

    return data, means_data, counts_data

# Data preparation for each density
val_scores_low = [[0.9597, 0.9674, 0.9568], 
                  [0.9674, 0.9044, 0.8523], 
                  [0.9658, 0.9052, 0.8936]]
test_scores_low = [[0.9437, 0.9282, 0.9568], 
                   [0.8984, 0.8850, 0.8523], 
                   [0.9006, 0.8096, 0.8367]]
val_counts_low = {'BE': np.mean([37.62, 36.70, 38.65]), 'NPV': np.mean([36.69, 37.09, 35.52]), 'PV': np.mean([25.68, 26.21, 25.82])}
test_counts_low = {'BE': np.mean([33.64, 39.88, 38.52]), 'NPV': np.mean([37.89, 35.32, 36.35]), 'PV': np.mean([28.47, 24.79, 25.13])}
data_low, means_data_low, counts_data_low = prepare_data(val_scores_low, test_scores_low, ['BE', 'NPV', 'PV'], val_counts_low, test_counts_low, 'LOW')

val_scores_med = [[0.9540, 0.9600, 0.9697], 
                  [0.7353, 0.7691, 0.7944], 
                  [0.9659, 0.9863, 0.9725], 
                  [0.8916, 0.9390, 0.9258]]
test_scores_med = [[0.9570, 0.9369, 0.7063], 
                   [0.7230, 0.5210, 0.6007], 
                   [0.9801, 0.9040, 0.8677], 
                   [0.9318, 0.7856, 0.9255]]
val_counts_med = {'BE': np.mean([34.21, 38.42, 37.01]), 'NPV': np.mean([5.56, 3.62, 4.29]), 'PV': np.mean([39.61, 35.88, 36.16]), 'SI': np.mean([20.62, 22.08, 22.54])}
test_counts_med = {'BE': np.mean([37.97, 33.30, 34.35]), 'NPV': np.mean([3.14, 5.82, 4.31]), 'PV': np.mean([36.89, 39.80, 36.84]), 'SI': np.mean([22.01, 21.08, 24.50])}
data_med, means_data_med, counts_data_med = prepare_data(val_scores_med, test_scores_med, ['BE', 'NPV', 'PV', 'SI'], val_counts_med, test_counts_med, 'MEDIUM')

val_scores_den = [[0.7589, 0.7548, 0.7303], 
                  [0.0000, 0.0000, 0.0000], 
                  [0.9241, 0.9192, 0.9160], 
                  [0.6842, 0.6618, 0.5903], 
                  [0.0000, 0.9803, 0.9720]]
test_scores_den = [[0.2599, 0.7641, 0.7472], 
                   [0.0000, 0.0000, 0.0000], 
                   [0.9166, 0.9101, 0.9129], 
                   [0.6345, 0.4722, 0.7035], 
                   [0.0000, 0.0000, 0.0000]]
val_counts_den = {'BE': np.mean([26.98, 18.36, 17.74]), 'NPV': np.mean([0.92, 1.87, 2.27]), 'PV': np.mean([53.23, 45.12, 44.60]), 'SI': np.mean([18.88, 12.79, 8.86]), 'WI': np.mean([0.00, 21.87, 26.52])}
test_counts_den = {'BE': np.mean([11.96, 30.86, 27.07]), 'NPV': np.mean([2.57, 0.97, 0.88]), 'PV': np.mean([36.34, 53.66, 51.19]), 'SI': np.mean([7.74, 14.51, 20.87]), 'WI': np.mean([41.40, 0.00, 0.00])}
data_den, means_data_den, counts_data_den = prepare_data(val_scores_den, test_scores_den, ['BE', 'NPV', 'PV', 'SI', 'WI'], val_counts_den, test_counts_den, 'DENSE')


# Combine Data for Faceted Plot
combined_data = pd.concat([data_low, data_med, data_den])
combined_means_data = pd.concat([means_data_low, means_data_med, means_data_den])
combined_counts_data = pd.concat([counts_data_low, counts_data_med, counts_data_den])


# Define the order of categories for Density
combined_data['Density'] = pd.Categorical(combined_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
combined_means_data['Density'] = pd.Categorical(combined_means_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
combined_counts_data['Density'] = pd.Categorical(combined_counts_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)

# Set the color scheme
class_color_scheme = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}
evaluation_set_colors = {'VALIDATION': '#55d400', 'TEST': '#5f00ff'}

# Plot with the specified order of facets
plot = (ggplot(combined_data, aes(x='Class', y='F1_Score'))
        + geom_violin(aes(fill='Class'), width=1, alpha=0.4)
        + geom_boxplot(aes(color='Evaluation set'), width=0.5, outlier_color='red', fill='white', alpha=1)
        + geom_point(data=combined_means_data, mapping=aes(x='Class', y='Mean_F1_Score', shape='Evaluation set'), 
                     color='black', size=4)
        + geom_text(data=combined_counts_data, mapping=aes(x='Class', y='y_position', label='Counts'), 
                    color='black', size=14, ha='center')
        + scale_fill_manual(values=class_color_scheme)  # Colors for violin plots by Class
        + scale_color_manual(values=evaluation_set_colors)  # Colors for boxplots by Evaluation set
        + labs(x='Class', y='F1-Score')
        + coord_cartesian(ylim=(0.0, 1.0))
        + facet_grid('~Density', scales='free_x', space='free_x')
        + theme(figure_size=(20, 12), 
                axis_title=element_text(size=18),
                axis_text=element_text(size=18),
                axis_text_x=element_text(size=18),
                axis_text_y=element_text(size=18),
                strip_text=element_text(size=18),  
                legend_title=element_text(size=18),
                legend_text=element_text(size=18),
                legend_position='right')
       )

# Save the plot to a file
plot.save("violin_boxplot_f1score_v1.png", dpi=300, width=20, height=12)
# Display the plot
print(plot)


#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_point, labs, theme, element_text, scale_fill_manual, scale_color_manual, coord_cartesian, facet_grid

# Prepare data function that works independently for each density
def prepare_data(val_scores, test_scores, classes, val_counts, test_counts, density_label):
    # Flatten scores for VALIDATION and TEST, aligning them by class
    validation_data = []
    for class_idx, class_name in enumerate(classes):
        validation_data.extend([(class_name, f1, "VALIDATION") for f1 in val_scores[class_idx]])

    test_data = []
    for class_idx, class_name in enumerate(classes):
        test_data.extend([(class_name, f1, "TEST") for f1 in test_scores[class_idx]])

    # Combine validation and test data into a DataFrame
    data = pd.DataFrame(
        validation_data + test_data,
        columns=["Class", "F1_Score", "Evaluation set"]
    )
    data["Density"] = density_label

    # Force categorical order
    data["Evaluation set"] = pd.Categorical(data["Evaluation set"], categories=["VALIDATION", "TEST"], ordered=True)
    data["Density"] = pd.Categorical(data["Density"], categories=["LOW", "MEDIUM", "DENSE"], ordered=True)

    # Calculate mean F1_Score for each class and evaluation set
    means_data = data.groupby(["Class", "Evaluation set"], as_index=False)["F1_Score"].mean()
    means_data.rename(columns={"F1_Score": "Mean_F1_Score"}, inplace=True)
    means_data["Density"] = density_label

    # Prepare counts data
    counts_data = pd.DataFrame({
        "Class": classes * 2,
        "Counts": [f'V= {val_counts[cls]:.2f}%' for cls in classes] +
                  [f'T= {test_counts[cls]:.2f}%' for cls in classes],
        "y_position": [1.03] * len(classes) + [1.00] * len(classes),
        "Evaluation set": ["VALIDATION"] * len(classes) + ["TEST"] * len(classes),
        "Density": [density_label] * 2 * len(classes)
    })
    counts_data["Density"] = pd.Categorical(counts_data["Density"], categories=["LOW", "MEDIUM", "DENSE"], ordered=True)

    return data, means_data, counts_data

# Data preparation for each density
val_scores_low = [[0.9597, 0.9674, 0.9568], 
                  [0.9674, 0.9044, 0.8523], 
                  [0.9658, 0.9052, 0.8936]]
test_scores_low = [[0.9437, 0.9282, 0.9568], 
                   [0.8984, 0.8850, 0.8523], 
                   [0.9006, 0.8096, 0.8367]]
val_counts_low = {'BE': np.mean([37.62, 36.70, 38.65]), 'NPV': np.mean([36.69, 37.09, 35.52]), 'PV': np.mean([25.68, 26.21, 25.82])}
test_counts_low = {'BE': np.mean([33.64, 39.88, 38.52]), 'NPV': np.mean([37.89, 35.32, 36.35]), 'PV': np.mean([28.47, 24.79, 25.13])}
data_low, means_data_low, counts_data_low = prepare_data(val_scores_low, test_scores_low, ['BE', 'NPV', 'PV'], val_counts_low, test_counts_low, 'LOW')

val_scores_med = [[0.9540, 0.9600, 0.9697], 
                  [0.7353, 0.7691, 0.7944], 
                  [0.9659, 0.9863, 0.9725], 
                  [0.8916, 0.9390, 0.9258]]
test_scores_med = [[0.9570, 0.9369, 0.7063], 
                   [0.7230, 0.5210, 0.6007], 
                   [0.9801, 0.9040, 0.8677], 
                   [0.9318, 0.7856, 0.9255]]
val_counts_med = {'BE': np.mean([34.21, 38.42, 37.01]), 'NPV': np.mean([5.56, 3.62, 4.29]), 'PV': np.mean([39.61, 35.88, 36.16]), 'SI': np.mean([20.62, 22.08, 22.54])}
test_counts_med = {'BE': np.mean([37.97, 33.30, 34.35]), 'NPV': np.mean([3.14, 5.82, 4.31]), 'PV': np.mean([36.89, 39.80, 36.84]), 'SI': np.mean([22.01, 21.08, 24.50])}
data_med, means_data_med, counts_data_med = prepare_data(val_scores_med, test_scores_med, ['BE', 'NPV', 'PV', 'SI'], val_counts_med, test_counts_med, 'MEDIUM')

val_scores_den = [[0.7589, 0.7548, 0.7303], 
                  [0.0000, 0.0000, 0.0000], 
                  [0.9241, 0.9192, 0.9160], 
                  [0.6842, 0.6618, 0.5903], 
                  [0.0000, 0.9803, 0.9720]]
test_scores_den = [[0.2599, 0.7641, 0.7472], 
                   [0.0000, 0.0000, 0.0000], 
                   [0.9166, 0.9101, 0.9129], 
                   [0.6345, 0.4722, 0.7035], 
                   [0.0000, 0.0000, 0.0000]]
val_counts_den = {'BE': np.mean([26.98, 18.36, 17.74]), 'NPV': np.mean([0.92, 1.87, 2.27]), 'PV': np.mean([53.23, 45.12, 44.60]), 'SI': np.mean([18.88, 12.79, 8.86]), 'WI': np.mean([0.00, 21.87, 26.52])}
test_counts_den = {'BE': np.mean([11.96, 30.86, 27.07]), 'NPV': np.mean([2.57, 0.97, 0.88]), 'PV': np.mean([36.34, 53.66, 51.19]), 'SI': np.mean([7.74, 14.51, 20.87]), 'WI': np.mean([41.40, 0.00, 0.00])}
data_den, means_data_den, counts_data_den = prepare_data(val_scores_den, test_scores_den, ['BE', 'NPV', 'PV', 'SI', 'WI'], val_counts_den, test_counts_den, 'DENSE')

# Combine Data for Faceted Plot
combined_data = pd.concat([data_low, data_med, data_den])
combined_means_data = pd.concat([means_data_low, means_data_med, means_data_den])
combined_counts_data = pd.concat([counts_data_low, counts_data_med, counts_data_den])

# Define the order of categories for Density
combined_data['Density'] = pd.Categorical(combined_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
combined_means_data['Density'] = pd.Categorical(combined_means_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
combined_counts_data['Density'] = pd.Categorical(combined_counts_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)

# Set the color scheme
class_color_scheme = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}
evaluation_set_colors = {'VALIDATION': '#55d400', 'TEST': '#5f00ff'}

# Create the plot
plot = (ggplot(combined_data, aes(x='Class', y='F1_Score'))
        + geom_violin(aes(fill='Class'), width=1, alpha=0.4)
        + geom_boxplot(aes(color='Evaluation set'), width=0.5, outlier_color='red', fill='white', alpha=1)
        + geom_point(data=combined_means_data, mapping=aes(x='Class', y='Mean_F1_Score', shape='Evaluation set'), 
                     color='black', size=4)
        + geom_text(data=combined_counts_data, mapping=aes(x='Class', y='y_position', label='Counts'), 
                    color='black', size=14, ha='center')
        + scale_fill_manual(values=class_color_scheme)  # Colors for violin plots by Class
        + scale_color_manual(values=evaluation_set_colors)  # Colors for boxplots by Evaluation set
        + labs(x='Class', y='F1-Score')
        + coord_cartesian(ylim=(0.0, 1.05))
        + facet_grid('~Density', scales='free_x', space='free_x')
        + theme(figure_size=(20, 12), 
                axis_title=element_text(size=18),
                axis_text=element_text(size=18),
                axis_text_x=element_text(size=18),
                axis_text_y=element_text(size=18),
                strip_text=element_text(size=18),  
                legend_title=element_text(size=18),
                legend_text=element_text(size=18),
                legend_position='right')
       )

# Save the plot to a file
plot.save("violin_boxplot_f1score_fixed.png", dpi=300, width=20, height=12)
# Display the plot
print(plot)



# %%
### version 2
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_point, labs, theme, element_text, scale_fill_manual, scale_color_manual, coord_cartesian, facet_grid, guide_legend, element_rect, theme_bw

# Define color schemes for Class and Evaluation set
class_color_scheme = {
    'BE': '#dae22f',
    'NPV': '#6332ea',
    'PV': '#e346ee',
    'SI': '#6da4d4',
    'WI': '#68e8d3'
}
boxplot_colors = {
    'VALIDATION': '#55d400',
    'TEST': '#5f00ff'
}

# Plot with the specified order of facets
plot = (ggplot(combined_data, aes(x='Class', y='F1_Score', fill='Class'))
        + geom_violin(aes(fill='Class'), width=1, alpha=0.3, show_legend=False)  # Use Class colors in violin plot
        + geom_boxplot(aes(color='Evaluation set'), width=0.5, outlier_color='red', fill='white', alpha=1)  # Boxplot with Evaluation set colors
        + geom_point(data=combined_means_data, 
                     mapping=aes(x='Class', y='Mean_F1_Score', shape='Evaluation set', fill='Evaluation set'),
                     color='black', size=4)  # Keep Evaluation set symbols in black
        + geom_text(data=combined_counts_data, mapping=aes(x='Class', y='y_position', label='Counts'), 
                    color='black', size=14, ha='center')
        
        # Apply color scales separately
        # + scale_fill_manual(
        #     name="Class",
        #     values={**class_color_scheme, **boxplot_colors},
        #     breaks=list(class_color_scheme.keys())  # Only show 'Class' items in the legend
        # )
        + scale_fill_manual(
            name="Class",
            values={**class_color_scheme, **boxplot_colors},
            breaks=list(class_color_scheme.keys()), # Only show 'Class' items in the legend
            guide=guide_legend(
                override_aes={'shape': 's', 'size':8}  # Use 's' for square (rectangle) shapes in the legend for Class
            )
        )
        + scale_color_manual(
            name="Evaluation set", 
            values=boxplot_colors,
            guide=guide_legend(
                # override_aes={'size':6}
            )  # Use guide_legend for proper legend control
        )     
        + labs(x='Class', y='F1-Score')
        + coord_cartesian(ylim=(0.0, 1.0))
        + facet_grid('~Density', scales='free_x', space='free_x')
        + theme_bw()  # Apply the black-and-white theme
        + theme(
            panel_background=element_rect(fill='white', color='black'),  # White panel background with a black border
            plot_background=element_rect(fill='white'),  # White plot background
            # panel_grid_major=element_line(color='lightgrey'),  # Dotted grid lines for major grids: linetype='dashed'
            # panel_grid_minor=element_line(color='lightgrey'),  # Dashed grid lines for minor grids
            strip_background=element_rect(fill='#FFF6E9', color='black'),  # Light blue background for facet grid labels: fill='black', color='white'
            figure_size=(20, 12), 
            axis_title=element_text(size=18),
            axis_text=element_text(size=18),
            axis_text_x=element_text(size=18),
            axis_text_y=element_text(size=18),
            strip_text=element_text(size=18),  
            legend_title=element_text(size=18),
            legend_text=element_text(size=18),  
            legend_key_size=20,
            legend_position='right',
            )
       )

# Save the plot to a file
plot.save("violin_boxplot_f1score_fixed.png", dpi=300, width=20, height=12)
# Display the plot
print(plot)





# %%
###################### ACROSS SITES  ##############################################################
## version 1: F1-Scores SITES
##test 5
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_point, labs, theme, element_text,
    scale_fill_manual, scale_color_manual, coord_cartesian
)

# Prepare data function
def prepare_data(val_scores, test_scores, classes, val_counts, test_counts, density_label):
    if len(val_scores) != len(classes) or len(test_scores) != len(classes):
        raise ValueError("The number of rows in val_scores and test_scores must match the number of classes.")

    # Flatten validation and test scores
    flattened_val_scores = []
    flattened_test_scores = []
    for class_idx in range(len(classes)):
        flattened_val_scores.extend(val_scores[class_idx])
        flattened_test_scores.extend(test_scores[class_idx])

    # Build DataFrame
    data = pd.DataFrame({
        'Class': classes * len(val_scores[0]) + classes * len(test_scores[0]),
        'F1_Score': flattened_val_scores + flattened_test_scores,
        'Evaluation set': ['VALIDATION'] * len(flattened_val_scores) +
                          ['TEST'] * len(flattened_test_scores),
        'Density': [density_label] * (len(flattened_val_scores) + len(flattened_test_scores))
    })

    # Calculate mean F1 scores for plotting
    means_data = data.groupby(['Class', 'Evaluation set'], as_index=False)['F1_Score'].mean()
    means_data.rename(columns={'F1_Score': 'Mean_F1_Score'}, inplace=True)
    means_data['Density'] = density_label

    # Prepare counts data
    counts_data = pd.DataFrame({
        'Class': classes * 2,
        'Counts': [f'V= {val_counts[cls]:.2f}%' for cls in classes] +
                  [f'T= {test_counts[cls]:.2f}%' for cls in classes],
        'y_position': [1.03] * len(classes) + [1.00] * len(classes),
        'Evaluation set': ['VALIDATION'] * len(classes) + ['TEST'] * len(classes),
        'Density': [density_label] * 2 * len(classes)
    })

    return data, means_data, counts_data


# Input data
test_scores_sites = [
    [0.0000, 0.0000, 0.3576],  # BE
    [0.0000, 0.0000, 0.0000],  # NPV
    [0.4200, 0.5410, 0.0000],  # PV
    [0.0000, 0.0000, 0.0000],  # SI
    [0.0000, 0.0000, 0.0000]   # WI
]

val_scores_sites = [
    [0.0000, 0.0000, 0.5603],  # BE
    [0.0000, 0.0000, 0.0000],  # NPV
    [0.5873, 0.5689, 0.0000],  # PV
    [0.0000, 0.0000, 0.0000],  # SI
    [0.0000, 0.0000, 0.0000]   # WI
]

# Classes
classes = ['BE', 'NPV', 'PV', 'SI', 'WI']

# Compute averages
val_averages = [np.mean(row) for row in val_scores_sites]
test_averages = [np.mean(row) for row in test_scores_sites]

# Link averages to classes
val_averages_by_class = {cls: avg for cls, avg in zip(classes, val_averages)}
test_averages_by_class = {cls: avg for cls, avg in zip(classes, test_averages)}

# Print averages for verification
print("Validation Averages by Class:")
for cls, avg in val_averages_by_class.items():
    print(f"{cls}: {avg:.3f}")

print("\nTest Averages by Class:")
for cls, avg in test_averages_by_class.items():
    print(f"{cls}: {avg:.3f}")

# Prepare data for plotting
data = pd.DataFrame({
    'Class': classes * 2,
    'F1_Score': val_averages + test_averages,
    'Evaluation set': ['VALIDATION'] * len(classes) + ['TEST'] * len(classes)
})

# Counts for each class
val_counts_sites = {
    'BE': np.mean([26.85, 26.00, 38.92]),
    'NPV': np.mean([3.33, 12.16, 17.41]),
    'PV': np.mean([26.59, 39.75, 30.83]),
    'SI': np.mean([17.59, 10.44, 12.84]),
    'WI': np.mean([10.65, 11.64, 0])
}

test_counts_sites = {
    'BE': np.mean([36.36, 36.61, 21.77]),
    'NPV': np.mean([37.05, 4.60, 21.77]),
    'PV': np.mean([26.59, 37.98, 45.47]),
    'SI': np.mean([0, 21.71, 13.36]),
    'WI': np.mean([0, 0, 17.25])
}

# Add counts to data
data['Counts'] = (
    [f"V= {val_counts_sites[cls]:.2f}%" for cls in classes] +
    [f"T= {test_counts_sites[cls]:.2f}%" for cls in classes]
)
data['y_position'] = [1.03] * len(classes) + [1.00] * len(classes)

# Define the color scheme
class_color_scheme = {'BE': '#dae22f', 'NPV': '#6332ea', 'PV': '#e346ee', 'SI': '#6da4d4', 'WI': '#68e8d3'}
evaluation_set_colors = {'VALIDATION': '#55d400', 'TEST': '#5f00ff'}

# Generate plot
plot = (ggplot(data, aes(x='Class', y='F1_Score'))
        + geom_violin(aes(fill='Class'), width=1, alpha=0.4)
        + geom_boxplot(aes(color='Evaluation set'), width=0.5, outlier_color='red', fill='white', alpha=1)
        + geom_point(aes(shape='Evaluation set'), color='black', size=4)
        + geom_text(aes(y='y_position', label='Counts'), color='black', size=10, ha='center')
        + scale_fill_manual(values=class_color_scheme)
        + scale_color_manual(values=evaluation_set_colors)
        + labs(x='Class', y='F1-Score', title='F1-Score Distribution for SITES Density')
        + coord_cartesian(ylim=(0.0, 1.0))
        + theme_bw()  # Apply the black-and-white theme
        + theme(
            panel_background=element_rect(fill='white', color='black'),  # White panel background with a black border
            plot_background=element_rect(fill='white'),  # White plot background
            # panel_grid_major=element_line(color='lightgrey'),  # Dotted grid lines for major grids: linetype='dashed'
            # panel_grid_minor=element_line(color='lightgrey'),  # Dashed grid lines for minor grids
            strip_background=element_rect(fill='#FFF6E9', color='black'),  # Light blue background for facet grid labels: fill='black', color='white'
            figure_size=(20, 12), 
            axis_title=element_text(size=18),
            axis_text=element_text(size=18),
            axis_text_x=element_text(size=18),
            axis_text_y=element_text(size=18),
            strip_text=element_text(size=18),  
            legend_title=element_text(size=18),
            legend_text=element_text(size=18),  
            legend_key_size=20,
            legend_position='right',
        )
    )

# Save and display the plot
plot.save("violin_boxplot_f1score_sites_corrected.png", dpi=300, width=20, height=12)
print(plot)
