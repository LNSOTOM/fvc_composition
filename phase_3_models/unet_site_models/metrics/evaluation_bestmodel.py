import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
import config_param
from torchmetrics.classification import ConfusionMatrix
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader, Subset

def initialize_all_metrics(num_blocks):
    """Initialize the metrics structure for all blocks."""
    return [{
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'miou': [],
        'unique_classes': [],
        'counts_per_class_percentage': [],
        'confusion_matrix': [],
        'all_preds_across_blocks': [],
        'all_trues_across_blocks': []
    } for _ in range(num_blocks)]

class ModelEvaluator:
    def __init__(self, model, data_loader, device=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=config_param.OUT_CHANNELS, ignore_index=-1).to(self.device)
        self.precision_metric = torchmetrics.Precision(task="multiclass", num_classes=config_param.OUT_CHANNELS, average='none', ignore_index=-1).to(self.device)
        self.recall_metric = torchmetrics.Recall(task="multiclass", num_classes=config_param.OUT_CHANNELS, average='none', ignore_index=-1).to(self.device)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=config_param.OUT_CHANNELS, average='none', ignore_index=-1).to(self.device)
        self.iou_metric = JaccardIndex(task="multiclass", num_classes=config_param.OUT_CHANNELS, ignore_index=-1, average='none').to(self.device)

    def evaluate(self):
        self.model.eval()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()
        self.iou_metric.reset()

        all_preds = []
        all_trues = []
        with torch.no_grad():
            for input_image, target_mask in self.data_loader:
                input_image, target_mask = input_image.to(self.device), target_mask.to(self.device)
                predicted_mask = self.model(input_image)
                
                # Ensure predicted_mask is of shape [batch_size, height, width]
                predicted_mask = torch.argmax(predicted_mask, dim=1)

                # Ensure target_mask is of shape [batch_size, height, width]
                if target_mask.ndim == 4:  # In case target_mask has an extra channel dimension
                    target_mask = target_mask.squeeze(1)

                valid_mask = target_mask != -1
                
                self.accuracy_metric(predicted_mask[valid_mask], target_mask[valid_mask])
                self.precision_metric(predicted_mask[valid_mask], target_mask[valid_mask])
                self.recall_metric(predicted_mask[valid_mask], target_mask[valid_mask])
                self.f1_metric(predicted_mask[valid_mask], target_mask[valid_mask])
                self.iou_metric(predicted_mask[valid_mask], target_mask[valid_mask])

                all_preds.extend(predicted_mask[valid_mask].cpu().numpy().flatten())
                all_trues.extend(target_mask[valid_mask].cpu().numpy().flatten())

        # Calculate unique classes across both predictions and targets
        unique_classes = np.unique(np.concatenate((all_preds, all_trues)))

        # Calculate confusion matrix
        conf_matrix_metric = ConfusionMatrix(task='multiclass', num_classes=len(unique_classes))
        conf_matrix = conf_matrix_metric(torch.as_tensor(all_preds), torch.as_tensor(all_trues))

        return {
            'accuracy': self.accuracy_metric.compute().item(),
            'precision': np.array(self.precision_metric.compute().cpu().numpy()),
            'recall': np.array(self.recall_metric.compute().cpu().numpy()),
            'f1': np.array(self.f1_metric.compute().cpu().numpy()),
            'iou': np.array(self.iou_metric.compute().cpu().numpy()),
            'miou': np.mean(self.iou_metric.compute().cpu().numpy()),
            'unique_classes': unique_classes,
            'counts_per_class_percentage': (np.bincount(all_trues, minlength=len(unique_classes)) / len(all_trues)) * 100,
            'confusion_matrix': conf_matrix.cpu().numpy(),
            'all_preds': all_preds,
            'all_trues': all_trues
        }

    def pad_confusion_matrix(self, conf_matrix, target_shape):
        """Pads a confusion matrix to the target shape with zeros."""
        padded_matrix = np.zeros(target_shape)
        padded_matrix[:conf_matrix.shape[0], :conf_matrix.shape[1]] = conf_matrix
        return padded_matrix


    #     return avg_metrics_across_blocks
    def calculate_average_metrics(self, all_metrics):
        # Updated method logic to handle missing or incomplete data
        valid_metrics = [m for m in all_metrics if m['accuracy']]  # Filter out empty or invalid metrics
        if not valid_metrics:
            print("No valid metrics to process.")
            return {}

        # Initialize accumulators for metrics
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        iou_dict = {}
        counts_dict = {}

        total_accuracy = sum(m['accuracy'] for m in valid_metrics)
        total_miou = sum(m['miou'] for m in valid_metrics)
        total_blocks = len(valid_metrics)

        for metrics in valid_metrics:
            unique_classes = metrics['unique_classes']
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            iou = metrics['iou']
            counts = metrics['counts_per_class_percentage']

            for idx, cls in enumerate(unique_classes):
                if cls not in precision_dict:
                    precision_dict[cls] = []
                    recall_dict[cls] = []
                    f1_dict[cls] = []
                    iou_dict[cls] = []
                    counts_dict[cls] = []

                precision_dict[cls].append(precision[idx])
                recall_dict[cls].append(recall[idx])
                f1_dict[cls].append(f1[idx])
                iou_dict[cls].append(iou[idx])
                counts_dict[cls].append(counts[idx])

        # Calculate averages
        avg_accuracy = total_accuracy / total_blocks
        avg_miou = total_miou / total_blocks

        avg_precision = {cls: np.mean(precision_dict[cls]) for cls in precision_dict}
        avg_recall = {cls: np.mean(recall_dict[cls]) for cls in recall_dict}
        avg_f1 = {cls: np.mean(f1_dict[cls]) for cls in f1_dict}
        avg_iou = {cls: np.mean(iou_dict[cls]) for cls in iou_dict}
        avg_counts = {cls: np.mean(counts_dict[cls]) for cls in counts_dict}

        # Prepare the results in the required format
        all_classes = sorted(precision_dict.keys())
        avg_precision_list = [avg_precision.get(cls, 0.0) for cls in all_classes]
        avg_recall_list = [avg_recall.get(cls, 0.0) for cls in all_classes]
        avg_f1_list = [avg_f1.get(cls, 0.0) for cls in all_classes]
        avg_iou_list = [avg_iou.get(cls, 0.0) for cls in all_classes]
        avg_counts_list = [avg_counts.get(cls, 0.0) for cls in all_classes]

        avg_metrics_across_blocks = {
            'accuracy': avg_accuracy,
            'precision': avg_precision_list,
            'recall': avg_recall_list,
            'f1': avg_f1_list,
            'iou': avg_iou_list,
            'miou': avg_miou,
            'unique_classes': all_classes,
            'counts_per_class_percentage': avg_counts_list
        }

        return avg_metrics_across_blocks



    def calculate_and_save_average_confusion_matrix(self, conf_matrices, all_preds, all_trues, class_labels, output_dir):
        unique_classes = sorted(set(all_preds) | set(all_trues))
        relevant_class_labels = [class_labels[i] for i in unique_classes]

        conf_matrices = [np.array(cm) for cm in conf_matrices if isinstance(cm, np.ndarray) and cm.ndim == 2]

        if not conf_matrices:
            print("No valid confusion matrices available after filtering.")
            return

        # Find the maximum shape for padding
        max_shape = (
            max(cm.shape[0] for cm in conf_matrices),
            max(cm.shape[1] for cm in conf_matrices)
        )

        # Pad all confusion matrices to the maximum shape
        padded_conf_matrices = [self.pad_confusion_matrix(cm, max_shape) for cm in conf_matrices]

        # Calculate the average confusion matrix
        avg_conf_matrix = np.mean(padded_conf_matrices, axis=0)

        # Normalize the average confusion matrix
        avg_conf_matrix_normalized = avg_conf_matrix / np.maximum(avg_conf_matrix.sum(axis=1, keepdims=True), 1)

        # Adjust class labels to the maximum size
        extended_class_labels = class_labels + ['Unknown'] * (max_shape[0] - len(class_labels))

        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot the averaged confusion matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(avg_conf_matrix_normalized, interpolation='nearest', cmap=sns.color_palette(palette='RdPu', as_cmap=True), vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(relevant_class_labels)))
        ax.set_xticklabels(relevant_class_labels, fontsize=18)
        ax.set_yticks(np.arange(len(relevant_class_labels)))
        ax.set_yticklabels(relevant_class_labels, fontsize=18)

        # Create a color bar to match the image height
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        colorbar.ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=12)

        # Annotate the confusion matrix with rounded values
        for i in range(avg_conf_matrix_normalized.shape[0]):
            for j in range(avg_conf_matrix_normalized.shape[1]):
                text = ax.text(j, i, f"{avg_conf_matrix_normalized[i, j]:.2%}",
                            ha="center", va="center", color="black", fontsize=16)

        plt.xlabel('True Labels', fontsize=20)
        plt.ylabel('Predicted Labels', fontsize=20)
        plt.title('Average Confusion Matrix Across All Blocks', fontsize=20)
        
        avg_cm_filename = os.path.join(output_dir, 'average_confusion_matrix_across_blocks.png')
        plt.savefig(avg_cm_filename, bbox_inches='tight')
        # plt.show()
        plt.close()

        print(f"Average confusion matrix across all blocks saved at {avg_cm_filename}")

    def plot_confusion_matrix(self, all_preds, all_trues, class_labels, output_dir, block_idx):
        # Ensure block_idx is treated as a string for concatenation
        block_idx_str = str(block_idx)

        unique_classes = sorted(set(all_preds) | set(all_trues))
        relevant_class_labels = [class_labels[i] for i in unique_classes]

        conf_matrix_metric = ConfusionMatrix(task='multiclass', num_classes=len(unique_classes))
        conf_matrix = conf_matrix_metric(torch.as_tensor(all_preds), torch.as_tensor(all_trues))

        # Normalize the confusion matrix to percentages
        conf_matrix_normalized = conf_matrix / np.maximum(conf_matrix.sum(axis=1, keepdims=True), 1)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(conf_matrix_normalized, interpolation='nearest', cmap=sns.color_palette(palette='RdPu', as_cmap=True), vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(relevant_class_labels)))
        ax.set_xticklabels(relevant_class_labels, fontsize=18)
        ax.set_yticks(np.arange(len(relevant_class_labels)))
        ax.set_yticklabels(relevant_class_labels, fontsize=18)

        # Create a color bar to match the image height
        colorbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        colorbar.ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=16)

        # Annotate the confusion matrix with values
        for i in range(conf_matrix_normalized.shape[0]):
            for j in range(conf_matrix_normalized.shape[1]):
                text = ax.text(j, i, f"{conf_matrix_normalized[i, j]:.2%}",
                            ha="center", va="center", color="black", fontsize=16)

        plt.xlabel('True Labels', fontsize=20)
        plt.ylabel('Predicted Labels', fontsize=20)
        # plt.title(f'Confusion Matrix {block_idx + 1}', fontsize=20)
        # plot_file = os.path.join(output_dir, f'block_{block_idx + 1}_confusion_matrix.png')
        plt.title(f'Confusion Matrix {block_idx_str}', fontsize=20)  # Use block_idx_str here
        plot_file = os.path.join(output_dir, f'block_{block_idx_str}_confusion_matrix.png')  # Use block_idx_str here as well

      
        plt.savefig(plot_file, bbox_inches='tight')
        # plt.show()
        plt.close()

        print(f"Confusion matrix plot saved as {plot_file}")

    def save_block_metrics(self, metrics, block, output_dir):
        block_metrics = (
            f"Block {block + 1} Metrics:\n"
            f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
            f"Precision: " + ", ".join([f"{p:.4f}" for p in metrics.get('precision', [])]) + "\n"
            f"Recall: " + ", ".join([f"{r:.4f}" for r in metrics.get('recall', [])]) + "\n"
            f"F1 Score: " + ", ".join([f"{f1s:.4f}" for f1s in metrics.get('f1', [])]) + "\n"
            f"IoU: " + ", ".join([f"{i:.4f}" for i in metrics.get('iou', [])]) + "\n"
            f"MIoU: {metrics.get('miou', 0):.4f}\n"
            f"Unique classes: " + ", ".join(map(str, metrics.get('unique_classes', []))) + "\n"
            f"Counts per class (%): " + ", ".join([f"{p:.2f}%" for p in metrics.get('counts_per_class_percentage', [])]) + "\n\n"
        )

        with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'a') as f:
            f.write(block_metrics)

        print(f"Metrics for Block {block + 1} saved to {output_dir}")

    def save_average_metrics(self, avg_metrics, block_idx, output_dir):
        try:
            if not avg_metrics:
                print("No average metrics to save.")
                return

            avg_metrics_text = (
                f"\nAverage Metrics Across All Blocks after Block {block_idx + 1}:\n"
                f"Accuracy: {avg_metrics['accuracy']:.4f}\n"
                f"Precision: " + ", ".join([f"{p:.4f}" for p in avg_metrics['precision']]) + "\n"
                f"Recall: " + ", ".join([f"{r:.4f}" for r in avg_metrics['recall']]) + "\n"
                f"F1 Score: " + ", ".join([f"{f1s:.4f}" for f1s in avg_metrics['f1']]) + "\n"
                f"IoU: " + ", ".join([f"{i:.4f}" for i in avg_metrics['iou']]) + "\n"
                f"MIoU: {avg_metrics['miou']:.4f}\n"
                f"Unique classes: " + ", ".join(map(str, avg_metrics['unique_classes'])) + "\n"
                f"Counts per class (%): " + ", ".join([f"{p:.2f}%" for p in avg_metrics['counts_per_class_percentage']]) + "\n\n"
            )

            with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'a') as f:
                f.write(avg_metrics_text)

            print(f"Average metrics across all blocks saved to {output_dir}")

        except KeyError as e:
            print(f"KeyError encountered: {e}")
            print(f"avg_metrics content: {avg_metrics}")
            raise

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def run_evaluation(self, block_idx, all_metrics, conf_matrices):
        # Initialize or append the all_preds_across_blocks and all_trues_across_blocks
        if block_idx == 0:
            all_preds_across_blocks = []
            all_trues_across_blocks = []
        else:
            # Fetch the already accumulated predictions and truths if not the first block
            all_preds_across_blocks = all_metrics[0].get('all_preds_across_blocks', [])
            all_trues_across_blocks = all_metrics[0].get('all_trues_across_blocks', [])

        metrics = self.evaluate()
        # block_output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/low' #low'
        # block_output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/medium' #medium
        # block_output_dir = '/media/laura/Extreme SSD/code/fvc_composition/phase_3_models/unet_model/outputs_ecosystems/dense' #dense
        block_output_dir = '/media/laura/Laura 102/fvc_composition/phase_3_models/unet_single_model/outputs_ecosystems/dense' #dense
        os.makedirs(block_output_dir, exist_ok=True)
        
        # Save metrics for the current block
        self.save_block_metrics(metrics, block_idx, block_output_dir)

        # Save the confusion matrix plot for this specific block
        self.plot_confusion_matrix(
            metrics['all_preds'], metrics['all_trues'], 
            class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'], 
            output_dir=block_output_dir, block_idx=block_idx
        )
        
        conf_matrices.append(metrics['confusion_matrix'])  # Store confusion matrix per block for later averaging
        
        # Aggregate predictions and ground truths across blocks
        all_preds_across_blocks.extend(metrics['all_preds'])
        all_trues_across_blocks.extend(metrics['all_trues'])
        
        # Store the accumulated predictions and truths back into all_metrics for future blocks
        all_metrics[0]['all_preds_across_blocks'] = all_preds_across_blocks
        all_metrics[0]['all_trues_across_blocks'] = all_trues_across_blocks

        # If all blocks have been processed, calculate and save the average metrics and confusion matrix across all blocks
        if block_idx == len(all_metrics) - 1:
            avg_metrics_across_blocks = self.calculate_average_metrics(all_metrics)
            self.save_average_metrics(avg_metrics_across_blocks, block_idx, block_output_dir)
            self.calculate_and_save_average_confusion_matrix(
                conf_matrices=conf_matrices,
                all_preds=all_preds_across_blocks,
                all_trues=all_trues_across_blocks,
                class_labels=['BE', 'NPV', 'PV', 'SI', 'WI'],
                output_dir=block_output_dir
            )

        return metrics