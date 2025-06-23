#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import rasterio
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# === Conditional Generator Model ===
class Generator(nn.Module):
    def __init__(self, latent_dim, mask_channels, output_shape):
        super(Generator, self).__init__()
        self.output_shape = output_shape
        self.mask_channels = mask_channels
        input_dim = latent_dim + mask_channels * np.prod(output_shape[1:])
        output_dim = torch.tensor(output_shape).prod().item()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z, mask):
        batch_size = z.size(0)
        mask_flat = mask.view(batch_size, -1)
        z_cond = torch.cat([z, mask_flat], dim=1)
        out = self.model(z_cond)
        return out.view(-1, *self.output_shape).float()

# === Discriminator Model ===
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        input_dim = torch.tensor(input_shape).prod().item()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# === Custom Dataset for NPV ===
class SimpleNPVDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
        self.images = []
        self.masks = []

        for img_path, msk_path in zip(self.image_paths, self.mask_paths):
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32) / np.finfo(src.dtypes[0]).max
                self.images.append(torch.from_numpy(img))

            with rasterio.open(msk_path) as src:
                msk = src.read(1).astype(np.int64)
                self.masks.append(torch.from_numpy(msk))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# === Instantiate Dataset from Provided Paths ===
original_dataset = SimpleNPVDataset(
    image_dir='/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/predictors_5b_subsample',
    mask_dir='/media/laura/Extreme SSD/qgis/calperumResearch/unet_model_5b/dense/mask_fvc_subsample'
)

# === Create a Copy to Augment with Synthetic Data in Memory ===
synthetic_dataset = deepcopy(original_dataset)

# === Training Loop (Conditional GAN) ===
def train_generative_model(generator, discriminator, dataset, latent_dim, device, epochs=10):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    adversarial_loss = nn.BCELoss()

    npv_pairs = [(img, mask) for img, mask in dataset if (mask == 1).any()]
    images = torch.stack([img for img, _ in npv_pairs]).to(device)
    masks = torch.stack([(mask == 1).float().unsqueeze(0) for _, mask in npv_pairs]).to(device)  # Binary mask for class 1

    for epoch in range(epochs):
        idx = torch.randint(0, len(images), (8,))
        real_imgs = images[idx]
        real_masks = masks[idx]
        z = torch.randn(8, latent_dim).to(device)
        fake_imgs = generator(z, real_masks).detach()

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        d_loss = adversarial_loss(real_validity, torch.ones_like(real_validity)) + \
                 adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        z = torch.randn(8, latent_dim).to(device)
        gen_imgs = generator(z, real_masks)
        validity = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, torch.ones_like(validity))

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# === Synthetic Sample Generation (conditioned on NPV mask) ===
def generate_synthetic_samples(generator, num_samples, latent_dim, device, mask_sample, original_shape):
    z = torch.randn(num_samples, latent_dim, device=device)
    masks = mask_sample.repeat(num_samples, 1, 1, 1).to(device)
    synthetic_images = generator(z, masks)
    synthetic_masks = torch.ones((num_samples, *original_shape[1:]), dtype=torch.long)
    return synthetic_images.cpu(), synthetic_masks.cpu()

# === Integrate into synthetic dataset (in memory only) ===
def integrate_synthetic_data(dataset, synthetic_images, synthetic_masks):
    dataset.images.extend([img.float() for img in synthetic_images])
    dataset.masks.extend([mask.long() for mask in synthetic_masks])
    print(f"✅ Added {len(synthetic_images)} synthetic NPV samples to the synthetic dataset in memory.")

#%%
# === Visualize Synthetic Samples ===
def visualize_synthetic_samples(synthetic_images, num=5):
    for i in range(min(num, len(synthetic_images))):
        img = synthetic_images[i].detach().numpy()  # Detach the tensor before calling .numpy()
        plt.figure(figsize=(6, 4))
        for b in range(img.shape[0]):
            plt.subplot(1, img.shape[0], b+1)
            plt.imshow(img[b], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Synthetic Sample {i+1}")
        plt.tight_layout()
        plt.show()

#%%
# === Example Execution ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(latent_dim=100, mask_channels=1, output_shape=synthetic_dataset.images[0].shape).to(device)
discriminator = Discriminator(input_shape=synthetic_dataset.images[0].shape).to(device)

#%%
train_generative_model(generator, discriminator, synthetic_dataset, latent_dim=100, device=device, epochs=10)

#%%
# use the first binary NPV mask as sample condition
sample_mask = (synthetic_dataset.masks[0] == 1).float().unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
synth_images, synth_masks = generate_synthetic_samples(generator, num_samples=5, latent_dim=100, device=device, mask_sample=sample_mask, original_shape=synthetic_dataset.images[0].shape)
integrate_synthetic_data(synthetic_dataset, synth_images, synth_masks)
visualize_synthetic_samples(synth_images)

# %

# %%
