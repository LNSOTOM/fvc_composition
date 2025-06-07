import torch
from dataset.data_augmentation import apply_combined_augmentations

class AugmentationWrapper(torch.utils.data.Dataset):
    """Wrapper to apply combined augmentations to the dataset."""
    def __init__(self, dataset):
        self.dataset = dataset

        # Expose the indices attribute if it exists in the wrapped dataset
        if hasattr(dataset, 'indices'):
            self.indices = dataset.indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # Apply combined augmentations
        image, mask = apply_combined_augmentations(image, mask)

        return image, mask

    def __getattr__(self, name):
        # Pass through any attributes not found to the underlying dataset
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")