
#%%
import numpy as np

# Training loss arrays for three blocks
block1 = np.array([0.9934, 0.8587, 0.7866, 0.7442, 0.7185, 0.6970, 0.6863, 0.6709, 0.6521, 0.6394, 0.6331, 0.6154, 0.6007, 0.5996, 0.5942, 0.5724, 0.5594, 0.5590, 0.5452, 0.5418, 0.5410, 0.5303, 0.5245, 0.5198, 0.5233, 0.5185, 0.5164, 0.5192, 0.5201, 0.5048, 0.5040, 0.5042, 0.5073, 0.5070, 0.5161, 0.5181, 0.5086, 0.5027, 0.5035, 0.5062, 0.5000, 0.5098, 0.5057, 0.5035, 0.5070, 0.5031, 0.5115, 0.5069, 0.5055, 0.5064, 0.5012, 0.5026, 0.4973, 0.4982, 0.4944, 0.5008, 0.5065, 0.5062, 0.4968, 0.5136, 0.5039, 0.5061, 0.4990, 0.4931, 0.5014, 0.5013, 0.5019, 0.4968, 0.4990, 0.4966, 0.4943, 0.4970, 0.4935, 0.5066, 0.5006, 0.5047, 0.4982, 0.4914, 0.4959, 0.4916, 0.4968, 0.5072, 0.5024, 0.4994, 0.5001, 0.4972, 0.4936, 0.4988, 0.4941, 0.4895, 0.4928, 0.4994, 0.4943, 0.4944, 0.4951, 0.4906, 0.4879, 0.4921, 0.4864, 0.4891, 0.4858, 0.5008, 0.4883, 0.4853, 0.4828, 0.4898, 0.4955, 0.4883, 0.4819, 0.4811, 0.4901, 0.4842, 0.4834, 0.4944, 0.4914, 0.4843, 0.4824, 0.4870, 0.4766, 0.4846
])

block2 = np.array([0.9066, 0.7707, 0.7184, 0.6762, 0.6491, 0.6311, 0.6162, 0.6007, 0.5735, 0.5689, 0.5601, 0.5416, 0.5477, 0.5371, 0.5301, 0.5170, 0.5168, 0.5140, 0.5070, 0.4952, 0.4907, 0.4857, 0.4910, 0.4962, 0.4922, 0.4835, 0.4795, 0.4811, 0.4852, 0.4761, 0.4772, 0.4805, 0.4826, 0.4837, 0.4787, 0.4856, 0.4696, 0.4707, 0.4769, 0.4830, 0.4776, 0.4773, 0.4716, 0.4711, 0.4775, 0.4793, 0.4733, 0.4807, 0.4717, 0.4688, 0.4639, 0.4749, 0.4651, 0.4710, 0.4703, 0.4692, 0.4667, 0.4735, 0.4801, 0.4728, 0.4687, 0.4631, 0.4651, 0.4657, 0.4709, 0.4664, 0.4741, 0.4602, 0.4810, 0.4629, 0.4668, 0.4745, 0.4761, 0.4720, 0.4684, 0.4631, 0.4730, 0.4693, 0.4731, 0.4746, 0.4728, 0.4679, 0.4691, 0.4610, 0.4705, 0.4686, 0.4656, 0.4602, 0.4640, 0.4650, 0.4629, 0.4654, 0.4549, 0.4585, 0.4620, 0.4628, 0.4684, 0.4614, 0.4623, 0.4637, 0.4603, 0.4642, 0.4543, 0.4584, 0.4588, 0.4712, 0.4535, 0.4596, 0.4645, 0.4581, 0.4649, 0.4681, 0.4547, 0.4563, 0.4518, 0.4513, 0.4527, 0.4552, 0.4561, 0.4583
])

block3 = np.array([0.8989, 0.7448, 0.6791, 0.6602, 0.6322, 0.6218, 0.6085, 0.5824, 0.5737, 0.5616, 0.5557, 0.5473, 0.5444, 0.5332, 0.5279, 0.5137, 0.5153, 0.5095, 0.4994, 0.5169, 0.4972, 0.5068, 0.4929, 0.4901, 0.4851, 0.4857, 0.5006, 0.4921, 0.4909, 0.4806, 0.4851, 0.4753, 0.4825, 0.4760, 0.4733, 0.4793, 0.4817, 0.4732, 0.4830, 0.4795, 0.4747, 0.4818, 0.4751, 0.4733, 0.4771, 0.4766, 0.4686, 0.4802, 0.4798, 0.4742, 0.4695, 0.4775, 0.4748, 0.4682, 0.4661, 0.4698, 0.4761, 0.4745, 0.4764, 0.4765, 0.4718, 0.4724, 0.4774, 0.4725, 0.4661, 0.4711, 0.4709, 0.4700, 0.4678, 0.4980, 0.4785, 0.4785, 0.4821, 0.4748, 0.4666, 0.4701, 0.4713, 0.4720, 0.4646, 0.4707, 0.4613, 0.4660, 0.4675, 0.4668, 0.4723, 0.4712, 0.4703, 0.4593, 0.4683, 0.4625, 0.4613, 0.4658, 0.4673, 0.4625, 0.4647, 0.4604, 0.4643, 0.4630, 0.4585, 0.4602, 0.4621, 0.4601, 0.4595, 0.4545, 0.4608, 0.4616, 0.4643, 0.4585, 0.4639, 0.4565, 0.4572, 0.4551, 0.4613, 0.4594, 0.4566, 0.4642, 0.4569, 0.4523, 0.4540, 0.4630
])

# Calculate the average training loss per epoch across blocks
average_training_loss = (block1 + block2 + block3) / 3

average_training_loss

# %%
import numpy as np
import pandas as pd

# Training loss arrays for three blocks
block1 = np.array([0.9815, 0.9730, 0.9610, 0.9561, 0.9508, 0.9441, 0.9423, 0.9403, 0.9380, 0.9300, 0.9206, 0.9107, 0.9175, 0.9279, 0.9060, 0.9061, 0.9049, 0.9010, 0.9014, 0.9188, 0.9059, 0.8999, 0.9329, 0.9075, 0.8987, 0.8970, 0.8991, 0.8970, 0.9174, 0.8975, 0.8973, 0.8976, 0.8961, 0.8968, 0.8968, 0, 1.0058, 0.8974, 0.9003, 0.9068, 0.8976, 0.9002, 0.8967, 0.8970, 0.8969, 0.8961, 0.8958, 0.8964, 0.8962, 0.8961, 0.8964, 0.8961, 0.8963, 0.8962, 0.8969, 0.8966, 0.8964, 0.8971, 0.8961, 0.8962, 0.8962, 0.8961, 0.8965, 0.8964, 0.8967, 0.8961, 0.8962, 0.8969, 0.8962, 0.8968, 0.8963, 0.8965, 0.8966, 0.8964, 0.8963, 0.8969, 0.8962, 0.8965, 0.8961, 0.8975, 0.8979, 0.8982, 0.8963, 0.8970, 0.8972, 0.8961, 0.8963, 0.8964, 0.8962, 0.8982, 0.8961, 0.8962, 0.8962, 0.8985, 0.8964, 0.8961, 0.8963, 0.8961, 0.8962, 0.8965, 0.8965, 0.8961, 0.8963, 0.8963, 0.8972, 0.8967, 0.8962, 0.8962, 0.8972, 0.8971, 0.8962, 0.8964, 0.8963, 0.8969, 0.8985, 0.8971, 0.8965, 0.8962, 0.8961, 0.8962
])

block2 = np.array([1.0123, 0.9845, 0.9899, 0.9740, 0.9805, 0.9544, 0.9488, 0.9730, 0.9671, 0, 0.9710, 0.9394, 0.9312, 0.9283, 0.9286, 0.9228, 0.9251, 0.9226, 0.9206, 0.9190, 0.9209, 0.9208, 0.9203, 0.9191, 0.9222, 0.9211, 0.9176, 0.9200, 0.9164, 0.9206, 0.9168, 0.9170, 0.9162, 0.9205, 0.9262, 0.9175, 0.9196, 0.9166, 0.9181, 0.9296, 0.9200, 0.9180, 0.9164, 0.9179, 0.9181, 0.9184, 0.9192, 0.9280, 0.9363, 0.9176, 0.9171, 0.9170, 0.9173, 0.9176, 0.9162, 0.9176, 0.9172, 0.9195, 0.9161, 0.9181, 0.9164, 0.9173, 0.9171, 0.9165, 0.9168, 0.9161, 0.9167, 0.9165, 0.9165, 0.9170, 0.9168, 0.9166, 0.9175, 0.9167, 0.9177, 0.9167, 0.9166, 0.9164, 0.9167, 0.9165, 0.9171, 0.9170, 0.9165, 0.9166, 0.9165, 0.9167, 0.9170, 0.9165, 0.9184, 0.9165, 0.9164, 0.9165, 0.9166, 0.9168, 0.9168, 0.9167, 0.9167, 0.9170, 0.9165, 0.9166, 0.9165, 0.9175, 0.9174, 0.9165, 0.9167, 0.9177, 0.9175, 0.9166, 0.9164, 0.9173, 0.9170, 0.9166, 0.9628, 0.9166, 0.9165, 0.9166, 0.9164, 0.9165, 0.9165, 0.9164
])

block3 = np.array([1.0030, 0.9871, 0.9786, 0.9643, 0.9533, 0.9444, 0.9369, 0.9311, 0.9264, 0.9224, 0.9190, 0.9163, 0.9410, 0.9121, 0.9102, 0.9067, 0.9058, 0.9051, 0.9094, 0.9038, 0.9032, 0.9024, 0.9033, 0.9050, 0.9026, 0.9020, 0.9021, 0.9014, 0.9021, 0.9014, 0.9011, 0.9024, 0.9009, 0.9002, 0.9009, 0.9008, 0.9005, 0.9030, 0.9015, 0.9000, 0.9027, 0.9001, 0.8998, 0.9009, 0.9031, 0.9014, 0.9023, 0.9015, 0.8997, 0.9010, 0.9012, 0.9006, 0.9027, 0.9002, 0.9004, 0.9007, 0.9040, 0.9031, 0.9013, 0.9028, 0.9012, 0.9035, 0.9051, 0.9018, 0.9022, 0.9003, 0.9025, 0.9012, 0.9012, 0.9026, 0.9037, 0.9014, 0.9025, 0.9008, 0.9046, 0.9005, 0.9011, 0.9012, 0.9016, 0.9044, 0.9017, 0.9015, 0.9011, 0.9020, 0.9016, 0.9032, 0.9010, 0.9029, 0.9010, 0.8998, 0.9016, 0.9000, 0.9002, 0.8994, 0.9030, 0.9013, 0.9023, 0.8997, 0.9003, 0.9019, 0.9003, 0.9018, 0.9032, 0.9015, 0.9003, 0.8999, 0.9007, 0.9022, 0.9016, 0.9049, 0.9029, 0.9018, 0.9027, 0.9020, 0.9010, 0.9007, 0.9013, 0.9012, 0.9010, 0.9011

])

# Calculate the average training loss per epoch across blocks
average_training_loss = (block1 + block2 + block3) / 3

# Create a DataFrame
epochs = np.arange(1, len(average_training_loss) + 1)
df = pd.DataFrame({'Epoch': epochs, 'Average_Training_Loss': average_training_loss})

# Save to CSV
df.to_csv('average_training_sites.csv', index=False)

print("Average training loss saved to 'average_training_sites.csv'")
df

# %%
import re
import numpy as np
import pandas as pd

# Function to extract losses from the text file
def extract_losses(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Extract training losses
    training_loss_matches = re.findall(r"Training Losses per Epoch: ([\d\., ]+)", text)
    training_losses = [list(map(float, match.split(", "))) for match in training_loss_matches]
    
    # Extract validation losses
    validation_loss_matches = re.findall(r"Validation Losses per Epoch: ([\d\., ]+)", text)
    validation_losses = [list(map(float, match.split(", "))) for match in validation_loss_matches]
    
    return training_losses, validation_losses

# Function to calculate average losses per epoch and save to CSV
def calculate_and_save_averages(training_losses, validation_losses, output_csv):
    # Calculate average training and validation losses per epoch
    num_epochs = len(training_losses[0])
    avg_training_losses = np.mean(training_losses, axis=0)
    avg_validation_losses = np.mean(validation_losses, axis=0)
    
    # Create a DataFrame
    epochs = np.arange(1, num_epochs + 1)
    df = pd.DataFrame({
        'Epoch': epochs,
        'Average_Training_Loss': avg_training_losses,
        'Average_Validation_Loss': avg_validation_losses
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Averages saved to {output_csv}")

# File paths
# For the original dataset
input_file = "/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense_original_with_water/loss_metrics.txt"  # Replace with your text file path
output_csv = "/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense_original_with_water/loss_metrics.csv"

## aug
# input_file = "/home/laura/dense_aug/loss_metrics.txt"  # Replace with your text file path
# output_csv = "/home/laura/dense_aug/loss_metrics.csv"

# Process the file and save results
training_losses, validation_losses = extract_losses(input_file)
calculate_and_save_averages(training_losses, validation_losses, output_csv)

# %%
df

# %%
###version 2
import re
import numpy as np
import pandas as pd

# Function to extract losses from the text file
def extract_losses(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Extract training and validation losses
    training_loss_matches = re.findall(r"Training Losses per Epoch: ([\d\., nan ]+)", text)
    validation_loss_matches = re.findall(r"Validation Losses per Epoch: ([\d\., nan ]+)", text)
    
    training_losses = []
    validation_losses = []
    
    for match in training_loss_matches:
        # Convert to float and handle 'nan'
        losses = [float(x) if x != 'nan' else np.nan for x in match.split(", ")]
        training_losses.append(losses)
    
    for match in validation_loss_matches:
        # Convert to float and handle 'nan'
        losses = [float(x) if x != 'nan' else np.nan for x in match.split(", ")]
        validation_losses.append(losses)
    
    return training_losses, validation_losses

# Function to calculate averages across blocks and save to CSV
def calculate_and_save_averages(training_losses, validation_losses, output_csv):
    # Find the minimum number of epochs across all blocks
    min_epochs_training = min(len(losses) for losses in training_losses)
    min_epochs_validation = min(len(losses) for losses in validation_losses)
    min_epochs = min(min_epochs_training, min_epochs_validation)
    
    # Trim losses to the minimum number of epochs
    training_losses_trimmed = [losses[:min_epochs] for losses in training_losses]
    validation_losses_trimmed = [losses[:min_epochs] for losses in validation_losses]
    
    # Calculate average losses per epoch, ignoring NaNs for validation
    avg_training_losses = np.nanmean(training_losses_trimmed, axis=0)
    avg_validation_losses = np.nanmean(validation_losses_trimmed, axis=0)
    
    # Create a DataFrame
    epochs = np.arange(1, min_epochs + 1)
    df = pd.DataFrame({
        'Epoch': epochs,
        'Average_Training_Loss': avg_training_losses,
        'Average_Validation_Loss': avg_validation_losses
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Averages saved to {output_csv}")

# File paths
#original
# input_file = "/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense_original_with_water/loss_metrics.txt"  # Replace with your text file path
# output_csv = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_original_with_water.csv"

#without water
input_file = "/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense_nowater/loss_metrics.txt"  # Replace with your text file path
output_csv = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_nowater.csv"

#without water + augmented
# input_file = "/home/laura/dense_aug/loss_metrics.txt"  # Replace with your text file path
# output_csv = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_aug.csv"

# Process the file and save results
try:
    training_losses, validation_losses = extract_losses(input_file)
    calculate_and_save_averages(training_losses, validation_losses, output_csv)
except Exception as e:
    print(f"Error: {e}")



# %%
###plot
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# Define the input CSV file and output directory
#original
# csv_file = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_original_with_water.csv"  # Replace with your text file path
# output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new"

#without water
csv_file = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_nowater.csv"  # Replace with your text file path
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new"

##without water + aug
# csv_file = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_aug.csv'  # Replace with the actual file path
# output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new'  # Replace with the actual output directory

# Load the data from the CSV file
loss_data = pd.read_csv(csv_file)

# Extract and round the loss values to 3 decimal places
epochs = loss_data['Epoch']
train_losses = loss_data['Average_Training_Loss'].round(3)
val_losses = loss_data['Average_Validation_Loss'].round(3)

# Adjust the x-axis to start from 0
x_lin = [0] + epochs.tolist()
train_losses_extended = [train_losses.iloc[0]] + [round(value, 3) for value in train_losses.tolist()]
val_losses_extended = [val_losses.iloc[0]] + [round(value, 3) for value in val_losses.tolist()]

# Plotting loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses_extended) + 1), train_losses_extended, color='#f54a19', label='Average Train Loss Across All Blocks')
plt.plot(range(1, len(val_losses_extended) + 1), val_losses_extended, color='blue', label='Average Validation Loss Across All Blocks')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
# Use a lighter color for the grid
plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# Force 3 decimal places on the y-axis
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

# Set the y-axis ticks with a range step of 0.025
y_min = min(train_losses_extended + val_losses_extended)
y_max = max(train_losses_extended + val_losses_extended)
plt.ylim(y_min, y_max)
plt.yticks([round(i * 0.125, 3) for i in range(int(y_min / 0.125), int(y_max / 0.125) + 2)])

# Save the plot
plt.savefig(os.path.join(output_dir, 'training_validation_loss_plot_low.png'), dpi=300)
plt.show()
print(f"Plot saved")



# %%
###plottt a
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# Define the datasets with their file paths
datasets = {
    "Low": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_low.csv",
    "Medium": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_medium.csv",
    "Dense": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_nowater.csv",
    # "Sites": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_sites.csv"
}

# Initialize variables to store global min and max values
global_min = float('inf')
global_max = float('-inf')

# Loop through datasets to find global min and max for consistent y-axis limits
for label, file_path in datasets.items():
    loss_data = pd.read_csv(file_path)
    train_losses = loss_data['Average_Training_Loss']
    val_losses = loss_data['Average_Validation_Loss']
    global_min = min(global_min, train_losses.min(), val_losses.min())
    global_max = max(global_max, train_losses.max(), val_losses.max())

# Add a buffer to y-axis limits
buffer = 0.05 * (global_max - global_min)  # 5% of the range as buffer
y_min = global_min - buffer
y_max = global_max + buffer

# Plot each dataset independently
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/plots"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

for label, file_path in datasets.items():
    loss_data = pd.read_csv(file_path)
    epochs = loss_data['Epoch']
    train_losses = loss_data['Average_Training_Loss'].round(3)
    val_losses = loss_data['Average_Validation_Loss'].round(3)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, color='#f54a19', label='Average Train Loss')
    plt.plot(epochs, val_losses, color='blue', label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Epoch vs. Average Loss - {label} Site')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.ylim(y_min, y_max)  # Apply consistent y-axis limits with buffer
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    # Save the plot
    output_path = os.path.join(output_dir, f"training_validation_loss_plot_{label.lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Plot saved for {label} Site at: {output_path}")


# %%
###plot b
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

# Define the datasets with their file paths
datasets = {
    "Low": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_low.csv",
    "Medium": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_medium.csv",
    "Dense": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/loss_averages_dense_nowater.csv",
    "Sites": "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/loss_averages_sites.csv"
}

# Initialize variables to store global min and max values
global_min = float('inf')
global_max = float('-inf')

# Loop through datasets to find global min and max for consistent y-axis limits
for label, file_path in datasets.items():
    loss_data = pd.read_csv(file_path)
    train_losses = loss_data['Average_Training_Loss']
    val_losses = loss_data['Average_Validation_Loss']
    global_min = min(global_min, train_losses.min(), val_losses.min())
    global_max = max(global_max, train_losses.max(), val_losses.max())

# Add a reduced buffer to y-axis limits
buffer = 0.03 * (global_max - global_min)  # 2% of the range as buffer
y_min = global_min - buffer
y_max = global_max + buffer

# Plot each dataset independently
output_dir = "/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/performance_metrics/new/plots"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

for label, file_path in datasets.items():
    loss_data = pd.read_csv(file_path)
    epochs = loss_data['Epoch']
    train_losses = loss_data['Average_Training_Loss'].round(3)
    val_losses = loss_data['Average_Validation_Loss'].round(3)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, color='#f54a19', label='Average Train Loss Across All Blocks')
    plt.plot(epochs, val_losses, color='blue', label='Average Validation Loss Across All Blocks')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    # plt.title(f'Epoch vs. Average Loss - {label} Site')
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.ylim(y_min, y_max)  # Apply consistent y-axis limits with reduced buffer
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    
    # Save the plot
    output_path = os.path.join(output_dir, f"training_validation_loss_plot_{label.lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Plot saved for {label} Site at: {output_path}")

# %%
