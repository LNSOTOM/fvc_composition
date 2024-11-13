
#%%
## IoU Scores
import pandas as pd
from plotnine import ggplot, aes, geom_violin, geom_boxplot, geom_text, geom_point, labs, theme, element_text, scale_fill_manual, scale_color_manual, coord_cartesian, facet_grid, guide_legend

# Prepare data function that works independently for each density
def prepare_data(val_scores, test_scores, classes, density_label):
    data = pd.DataFrame({
        'Class': classes * 6,
        'IoU': sum(val_scores, []) + sum(test_scores, []),
        'Evaluation set': ['VALIDATION'] * 3 * len(classes) + ['TEST'] * 3 * len(classes),
        'Density': [density_label] * 6 * len(classes)
    })
    
    # Set order for 'Evaluation set' and 'Density' categories
    data['Evaluation set'] = pd.Categorical(data['Evaluation set'], categories=['VALIDATION', 'TEST'], ordered=True)
    data['Density'] = pd.Categorical(data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
    
    # Calculate means for each class and evaluation set
    means_data = data.groupby(['Class', 'Evaluation set'], as_index=False)['IoU'].mean()
    means_data.rename(columns={'IoU': 'Mean_IoU'}, inplace=True)
    means_data['Density'] = density_label

    return data, means_data

# Prepare data for each density level
data_low, means_data_low = prepare_data(
    [[0.9225, 0.7891, 0.7052], [0.9369, 0.8254, 0.8064], [0.9338, 0.8268, 0.8077]],
    [[0.8933, 0.8156, 0.8191], [0.8661, 0.7937, 0.7425], [0.9172, 0.6801, 0.7192]],
    ['BE', 'NPV', 'PV'],
    'LOW'
)

data_medium, means_data_medium = prepare_data(
    [[0.9121, 0.5814, 0.9340, 0.8043], [0.9231, 0.6248, 0.9729, 0.8851], [0.9411, 0.6590, 0.9466, 0.8618]],
    [[0.9175, 0.5661, 0.9610, 0.8723], [0.8813, 0.3523, 0.8247, 0.6469], [0.5460, 0.4292, 0.7663, 0.8613]],
    ['BE', 'NPV', 'PV', 'SI'],
    'MEDIUM'
)

data_dense, means_data_dense = prepare_data(
    [[0.6115, 0.0000, 0.8589, 0.5200, 0.0000], [0.6061, 0.0000, 0.8506, 0.4945, 0.9613], [0.5752, 0.0000, 0.8451, 0.4188, 0.9455]],
    [[0.1494, 0.0000, 0.8460, 0.4647, 0.0000], [0.6183, 0.0000, 0.8350, 0.3091, 0.0000], [0.5964, 0.0000, 0.8397, 0.5427, 0.0000]],
    ['BE', 'NPV', 'PV', 'SI', 'WI'],
    'DENSE'
)

# Combine data for plotting
combined_data = pd.concat([data_low, data_medium, data_dense])
combined_means_data = pd.concat([means_data_low, means_data_medium, means_data_dense])


# Combine data for plotting
combined_data['Density'] = pd.Categorical(combined_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)
combined_means_data['Density'] = pd.Categorical(combined_means_data['Density'], categories=['LOW', 'MEDIUM', 'DENSE'], ordered=True)

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

# Create the plot
plot = (ggplot(combined_data, aes(x='Class', y='IoU', fill='Evaluation set'))
        # + geom_boxplot(position='dodge', alpha=0.5)
        + geom_boxplot(aes(color='Evaluation set'), width=0.5, outlier_color='red', fill='white', alpha=1)  # Boxplot with Evaluation set colors
        + geom_point(data=combined_means_data, mapping=aes(x='Class', y='Mean_IoU', shape='Evaluation set'), color='black', size=4)
        + scale_fill_manual(values=['#55d400', '#5f00ff'])  # Colors for Validation and Test sets
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
        + labs(x='Class', y='IoU', title='')
        + facet_grid('.~Density', scales='free_x', space='free_x')  # Free x-scales for each density and enforce facet order
        + theme(figure_size=(14, 8),
                axis_text=element_text(size=12),
                axis_title=element_text(size=14),
                strip_text=element_text(size=14),
                legend_title=element_text(size=12),
                legend_position='right'))


# Save the plot
plot.save("boxplot_iou_scores.png", dpi=300, width=14, height=8)

# Display the plot
print(plot)
# %%
