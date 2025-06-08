import os
import time
import torch
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import psutil

# Utility function to print GPU memory usage
def print_gpu_memory_usage(stage="", reset_peak=False):
    """Enhanced memory tracking with option to reset peak memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        print(f"{stage} - GPU Memory: "
              f"Current={allocated:.2f}GB, "
              f"Reserved={reserved:.2f}GB, "
              f"Peak={max_allocated:.2f}GB")
        
        # Reset peak memory stats if requested
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
            
    # CPU memory
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)
    print(f"{stage} - RAM Usage: {ram_usage:.2f}GB")

def run_training_loop(model, train_loader, val_loader, optimizer, criterion, max_epochs, block_idx, output_dir, device='cpu', logger=None, accumulation_steps=2):
    train_losses_per_epoch = []
    val_losses_per_epoch = []
    best_val_loss = float('inf')
    best_model_path = None
    
    os.makedirs(output_dir, exist_ok=True)

    if logger:
        writer = SummaryWriter(log_dir=logger.log_dir)
    else:
        writer = SummaryWriter()

    model.to(device)
    scaler = GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=max_epochs)
    # scheduler = OneCycleLR(
    #     optimizer, 
    #     max_lr=5e-4,  # Reduced from 1e-3
    #     steps_per_epoch=len(train_loader), 
    #     epochs=max_epochs,
    #     pct_start=0.2,  # Start with warmup phase
    #     div_factor=10,  # Start at lr/10 instead of lr/25
    #     final_div_factor=1000  # End at lr/1000 instead of lr/10000
    # )

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

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss_sum += loss.item() * accumulation_steps
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Iteration {batch_idx+1} - Loss: {loss.item():.4f}')
                print_gpu_memory_usage(f"Training - Epoch {epoch+1}, Step {batch_idx+1}")
            
            torch.cuda.empty_cache()
            del images, masks, outputs, loss
            gc.collect()

        avg_train_loss = train_loss_sum / num_batches
        train_losses_per_epoch.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss_sum += loss.item()
                num_val_batches += 1
                
                torch.cuda.empty_cache()
                del images, masks, outputs, loss
                gc.collect()

        avg_val_loss = val_loss_sum / num_val_batches
        val_losses_per_epoch.append(avg_val_loss)

        # Logging to TensorBoard
        if logger:
            logger.log_metrics({'train_loss': avg_train_loss, 'val_loss': avg_val_loss}, epoch)

        print(f"Epoch [{epoch+1}/{max_epochs}] - Average Train Loss: {avg_train_loss:.4f}, Average Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        
        print_gpu_memory_usage(f"End of Epoch {epoch+1}")

        # Save model checkpoint after each epoch
        epoch_model_path = os.path.join(output_dir, f'block_{block_idx + 1}_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model saved at {epoch_model_path}")

        # Update the best model if current validation loss is the lowest
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = epoch_model_path

    total_duration = time.time() - total_start_time
    print(f"Total training time for block {block_idx + 1}: {total_duration:.2f} seconds")

    # Plotting loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_losses_per_epoch, color='#fc3468', label='Average Train Loss')
    plt.plot(range(1, max_epochs + 1), val_losses_per_epoch, color='blue', label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(f'Epoch vs. Average Loss - Block {block_idx + 1}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'block_{block_idx + 1}_training_validation_loss_plot.png'))
    plt.show()
    print(f"Plot saved")

    writer.close()
    
    print(f"Best model saved at {best_model_path} with validation loss: {best_val_loss:.4f}")
    return train_losses_per_epoch, val_losses_per_epoch, best_model_path, best_val_loss
