import os
import time
import torch
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import psutil

# Utility function to print GPU memory usage
# def print_gpu_memory_usage(stage="", reset_peak=False):
#     """Enhanced memory tracking with option to reset peak memory usage."""
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / (1024 ** 3)
#         reserved = torch.cuda.memory_reserved() / (1024 ** 3)
#         max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
#         print(f"{stage} - GPU Memory: "
#               f"Current={allocated:.2f}GB, "
#               f"Reserved={reserved:.2f}GB, "
#               f"Peak={max_allocated:.2f}GB")
        
#         # Reset peak memory stats if requested
#         if reset_peak:
#             torch.cuda.reset_peak_memory_stats()
            
#     # CPU memory
#     process = psutil.Process(os.getpid())
#     ram_usage = process.memory_info().rss / (1024 ** 3)
#     print(f"{stage} - RAM Usage: {ram_usage:.2f}GB")

def print_gpu_memory_usage(stage=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert bytes to GB
    print(f"{stage} - Allocated memory: {allocated:.2f} GB, Cached memory: {cached:.2f} GB")


def run_training_loop(model, train_loader, val_loader, optimizer, criterion, max_epochs, block_idx, output_dir, device='cpu', logger=None, accumulation_steps=2):
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    best_val_loss = float('inf')
    best_model_path = None

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=logger.log_dir if logger else None)

    model.to(device)
    scaler = GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=max_epochs)

    total_start_time = time.time()

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0
        num_batches = 0
        optimizer.zero_grad()

        print_gpu_memory_usage(f"Start of Epoch {epoch+1}")

        for batch_idx, batch in enumerate(train_loader):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            with autocast(device_type=device):
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps

            if not torch.isfinite(loss):
                print(f"⚠️ Skipping non-finite loss at Epoch {epoch+1}, Batch {batch_idx+1}")
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss_sum += loss.item() * accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Iteration {batch_idx+1} - Loss: {loss.item():.4f}')
                print_gpu_memory_usage(f"Training - Epoch {epoch+1}, Step {batch_idx+1}")

            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        avg_train_loss = train_loss_sum / max(1, num_batches)
        train_losses_per_epoch.append(avg_train_loss)

        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                with autocast(device_type=device):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                if torch.isfinite(loss):
                    val_loss_sum += loss.item()
                    num_val_batches += 1

                del images, masks, outputs, loss
                torch.cuda.empty_cache()
                gc.collect()

        avg_val_loss = val_loss_sum / max(1, num_val_batches)
        val_losses_per_epoch.append(avg_val_loss)

        if logger:
            logger.log_metrics({'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, epoch)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Memory/Allocated_GB', torch.cuda.memory_allocated() / 1e9, epoch)
        writer.add_scalar('Memory/Reserved_GB', torch.cuda.memory_reserved() / 1e9, epoch)

        print(f"Epoch [{epoch+1}/{max_epochs}] - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        print_gpu_memory_usage(f"End of Epoch {epoch+1}")

        epoch_model_path = os.path.join(output_dir, f'block_{block_idx + 1}_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved at {epoch_model_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = epoch_model_path

        torch.cuda.empty_cache()
        gc.collect()

    total_duration = time.time() - total_start_time
    print(f"Total training time for block {block_idx + 1}: {total_duration:.2f} seconds")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_losses_per_epoch, color='#fc3468', label='Average Train Loss')
    plt.plot(range(1, max_epochs + 1), val_losses_per_epoch, color='blue', label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Epoch vs. Average Loss - Block {block_idx + 1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'block_{block_idx + 1}_training_validation_loss_plot.png'))
    print("Plot saved")

    writer.close()
    print(f"Best model saved at {best_model_path} with validation loss: {best_val_loss:.4f}")
    return train_losses_per_epoch, val_losses_per_epoch, best_model_path, best_val_loss
