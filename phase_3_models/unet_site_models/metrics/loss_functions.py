import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        # Create a mask to ignore specified indices (e.g., NaN or background)
        valid_mask = (targets != self.ignore_index)

        # Flatten the targets and outputs for masking and computation
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))  # Shape: [batch_size * height * width, num_classes]
        targets = targets.view(-1)  # Shape: [batch_size * height * width]

        # Apply the valid mask
        outputs = outputs[valid_mask.view(-1)]
        # targets = targets[valid_mask.view(-1)]
        targets = targets[valid_mask.view(-1)].long() 

        # Convert targets to one-hot encoding
        num_classes = outputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Apply softmax to outputs to get class probabilities
        outputs = torch.softmax(outputs, dim=1)

        # Compute intersection and union
        intersection = (outputs * targets_one_hot).sum(dim=0)
        dice_denominator = outputs.sum(dim=0) + targets_one_hot.sum(dim=0)

        # Compute Dice score
        dice_score = (2. * intersection + self.smooth) / (dice_denominator + self.smooth)

        # Dice loss is 1 minus the Dice score
        dice_loss = 1 - dice_score.mean()

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        # Create a mask to ignore NaN values (or the specified ignore_index)
        valid_mask = (targets != self.ignore_index)

        # Flatten the tensors to apply the mask
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, outputs.size(1))  # Shape: [batch_size * height * width, num_classes]
        targets = targets.view(-1)  # Shape: [batch_size * height * width]

        # Apply the valid mask
        outputs = outputs[valid_mask.view(-1)]
        targets = targets[valid_mask.view(-1)]

        # Compute the cross entropy loss
        logpt = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-logpt)

        # Apply Focal Loss formula
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * logpt

        return focal_loss.mean()

'''
The alpha parameter controls the weighting between Focal Loss and Dice Loss in the combined loss function.
alpha=0.8: This gives 80% of the total loss weight to Focal Loss and 20% to Dice Loss. 
This setup is useful if you find that Focal Loss is generally performing better, 
but you still want to retain some influence from Dice Loss, 
especially for cases involving boundary accuracy or small objects.
'''
class CombinedDiceFocalLoss(nn.Module):  #alpha=0.5
    def __init__(self, alpha=0.8, gamma=2, smooth=1, ignore_index=-1):
        super(CombinedDiceFocalLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=1, gamma=gamma, ignore_index=ignore_index)
        self.alpha = alpha

    def forward(self, outputs, targets):
        dice_loss = self.dice_loss(outputs, targets)
        focal_loss = self.focal_loss(outputs, targets)
        combined_loss = self.alpha * focal_loss + (1 - self.alpha) * dice_loss
        return combined_loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        
        intersection = (outputs * targets).sum(dim=2).sum(dim=2)
        union = outputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, device='cpu', ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.device = device
        self.weights = weights.to(self.device) if weights is not None else None
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        if self.weights is not None:
            self.weights = self.weights.to(outputs.device)
        loss = nn.CrossEntropyLoss(weight=self.weights, ignore_index=self.ignore_index)(outputs, targets)
        return loss

# Calculate class weights based on class frequencies
def calculate_class_weights(class_frequencies, device):
    class_weights = 1.0 / class_frequencies
    class_weights = class_weights / class_weights.sum() * len(class_frequencies)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# Save the calculated class weights to a text file
def save_class_weights_to_file(class_weights, file_path):
    with open(file_path, 'w') as file:
        for weight in class_weights.cpu().numpy():
            file.write(f"{weight}\n")