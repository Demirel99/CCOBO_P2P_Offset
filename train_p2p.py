#train_p2p.py
"""
Iterative Crowd Counting Model Training Script (Offset Regression Hybrid)
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Or ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import json 

from config_p2p import (
    DEVICE, SEED, TOTAL_ITERATIONS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    VALIDATION_INTERVAL, VALIDATION_BATCHES, SCHEDULER_PATIENCE, 
    IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, OUTPUT_DIR, LOG_FILE_PATH, BEST_MODEL_PATH,
    AUGMENTATION_SIZE, MODEL_INPUT_SIZE, GT_PSF_SIGMA,
    HEAD_FPN_OUTPUT_STRIDE # For anchor generation during validation
)
from utils import set_seed, find_and_sort_paths, split_train_val, generate_anchor_grid_centers
from dataset_p2p import generate_batch, generate_train_sample, ANCHOR_POINTS_NORM # Import ANCHOR_POINTS_NORM for validation
from model_p2p import VGG19FPNASPP
from losses_p2p import combined_offset_regression_loss

# --- Pre-generate anchor coordinates (normalized) for validation ---
# This ANCHOR_POINTS_NORM_VAL is the same as ANCHOR_POINTS_NORM in dataset_p2p.py
# but we make it explicit here for clarity in the training script.
_FEATURE_MAP_H_VAL = MODEL_INPUT_SIZE // HEAD_FPN_OUTPUT_STRIDE
_FEATURE_MAP_W_VAL = MODEL_INPUT_SIZE // HEAD_FPN_OUTPUT_STRIDE
ANCHOR_POINTS_NORM_VAL = generate_anchor_grid_centers(_FEATURE_MAP_H_VAL, _FEATURE_MAP_W_VAL).to(DEVICE)


def log_hyperparameters(log_file_path, config_module):
    """Logs hyperparameters from the config module to the log file."""
    hyperparams = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if not key.startswith("__") and not callable(getattr(config_module, key))
        and isinstance(getattr(config_module, key), (int, float, str, list, dict, tuple, bool, torch.device)) 
    }
    if 'DEVICE' in hyperparams and isinstance(hyperparams['DEVICE'], torch.device):
        hyperparams['DEVICE'] = str(hyperparams['DEVICE'])
        
    with open(log_file_path, "a") as log_file:
        log_file.write("--- Hyperparameters ---\n")
        log_file.write(json.dumps(hyperparams, indent=4))
        log_file.write("\n--- Training Log ---\n")

def train():
    print("Setting up training (Offset Regression Mode)...")
    set_seed(SEED)

    if os.path.exists(LOG_FILE_PATH):
        try: os.remove(LOG_FILE_PATH)
        except OSError as e: print(f"Warning: Could not remove existing log file: {e}")
    
    import config_p2p as cfg # Import config module to pass to log_hyperparameters
    log_hyperparameters(LOG_FILE_PATH, cfg)


    sorted_image_paths_train_val = find_and_sort_paths(IMAGE_DIR_TRAIN_VAL, '*.jpg')
    sorted_gt_paths_train_val = find_and_sort_paths(GT_DIR_TRAIN_VAL, '*.mat') 
    if not sorted_gt_paths_train_val:
        print("Warning: GT paths list is empty.")
        sorted_gt_paths_train_val = [None] * len(sorted_image_paths_train_val)

    if not sorted_image_paths_train_val:
        raise FileNotFoundError("Training/Validation images not found. Check paths in config.py.")

    train_image_paths, train_gt_paths, val_image_paths, val_gt_paths = split_train_val(
        sorted_image_paths_train_val, sorted_gt_paths_train_val, val_ratio=0.1, seed=SEED
    )
    if not train_image_paths or not val_image_paths:
        raise ValueError("Train or validation set is empty after splitting.")

    model = VGG19FPNASPP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_ITERATIONS, eta_min=1e-6) 
    print(f"Using CosineAnnealingLR scheduler with T_max={TOTAL_ITERATIONS}, eta_min=1e-6.")
    
    criterion = combined_offset_regression_loss 
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp: print("Using Automatic Mixed Precision (AMP).")

    best_val_loss = float('inf') # Here, val_loss is the combined loss for consistency. Pixel error is the key metric.
    best_val_pixel_error = float('inf')
    iterations_list, train_loss_list, val_loss_list_combined, val_pixel_error_list = [], [], [], []


    print("Starting training...")
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc=f"Iteration 1/{TOTAL_ITERATIONS}", unit="iter")
    train_loss_accum, train_cls_loss_accum, train_reg_loss_accum, samples_in_accum = 0.0, 0.0, 0.0, 0

    for iteration in pbar:
        model.train()
        # generate_batch now returns: img, psf, target_offsets, target_labels, direct_coords_for_eval
        img_batch, in_psf_batch, tgt_offsets_batch, tgt_labels_batch, _ = generate_batch(
            train_image_paths, train_gt_paths, BATCH_SIZE,
            generation_fn=generate_train_sample,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA
        )

        if img_batch is None:
            print(f"Warning: Failed to generate training batch at iter {iteration}. Skipping.")
            if scheduler: scheduler.step() 
            continue

        img_batch = img_batch.to(DEVICE)
        in_psf_batch = in_psf_batch.to(DEVICE)
        tgt_offsets_batch = tgt_offsets_batch.to(DEVICE)
        tgt_labels_batch = tgt_labels_batch.to(DEVICE)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            predicted_offsets, predicted_logits = model(img_batch, in_psf_batch)
            loss, cls_loss, reg_loss = criterion(predicted_offsets, predicted_logits, tgt_offsets_batch, tgt_labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        train_loss_accum += loss.item() * img_batch.size(0)
        train_cls_loss_accum += cls_loss.item() * img_batch.size(0)
        train_reg_loss_accum += reg_loss.item() * img_batch.size(0)
        samples_in_accum += img_batch.size(0)

        if iteration == 1 and BATCH_SIZE > 0: # First batch printout
            print("\n--- First Training Batch Output Sample (Normalized [0,1]) ---")
            # Get one predicted coord
            probs_first_batch = torch.sigmoid(predicted_logits.detach()) # (B, N_anchors, 1)
            best_anchor_idx_b = torch.argmax(probs_first_batch.squeeze(-1), dim=1) # (B,)
            
            pred_offset_samp = predicted_offsets.detach()[0, best_anchor_idx_b[0], :].cpu().numpy()
            pred_anchor_coord_samp = ANCHOR_POINTS_NORM_VAL[best_anchor_idx_b[0]].cpu().numpy()
            pred_final_coord_samp = pred_anchor_coord_samp + pred_offset_samp
            
            # Get corresponding target
            gt_positive_idx_b = torch.argmax(tgt_labels_batch.squeeze(-1), dim=1) # (B,)
            gt_offset_samp = tgt_offsets_batch.detach()[0, gt_positive_idx_b[0], :].cpu().numpy()
            gt_anchor_coord_samp = ANCHOR_POINTS_NORM_VAL[gt_positive_idx_b[0]].cpu().numpy()
            gt_final_coord_samp = gt_anchor_coord_samp + gt_offset_samp

            print(f"  Sample 0 Pred Final: [{pred_final_coord_samp[0]:.3f}, {pred_final_coord_samp[1]:.3f}] "
                  f"(Anchor: [{pred_anchor_coord_samp[0]:.3f}, {pred_anchor_coord_samp[1]:.3f}], Offset: [{pred_offset_samp[0]:.3f}, {pred_offset_samp[1]:.3f}])")
            print(f"  Sample 0 GT   Final: [{gt_final_coord_samp[0]:.3f}, {gt_final_coord_samp[1]:.3f}] "
                  f"(Anchor: [{gt_anchor_coord_samp[0]:.3f}, {gt_anchor_coord_samp[1]:.3f}], Offset: [{gt_offset_samp[0]:.3f}, {gt_offset_samp[1]:.3f}])")
            print("------------------------------------------------------------")


        if iteration % VALIDATION_INTERVAL == 0:
            avg_train_loss = train_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            avg_train_cls_loss = train_cls_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0
            avg_train_reg_loss = train_reg_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0

            rng_state = {'random': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                         'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
            
            model.eval()
            total_val_loss_combined, total_val_cls_loss, total_val_reg_loss, total_val_samples = 0.0, 0.0, 0.0, 0
            total_val_pixel_error = 0.0 
            printed_val_coords_this_interval = False

            with torch.no_grad():
                for val_batch_idx in range(VALIDATION_BATCHES):
                    set_seed(SEED + iteration + val_batch_idx) 
                    
                    val_img, val_in_psf, val_tgt_offsets, val_tgt_labels, val_direct_tgt_coords = generate_batch(
                        val_image_paths, val_gt_paths, BATCH_SIZE,
                        generation_fn=generate_train_sample,
                        augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA
                    )
                    if val_img is None: continue
                    val_img, val_in_psf = val_img.to(DEVICE), val_in_psf.to(DEVICE)
                    val_tgt_offsets, val_tgt_labels = val_tgt_offsets.to(DEVICE), val_tgt_labels.to(DEVICE)
                    val_direct_tgt_coords = val_direct_tgt_coords.to(DEVICE) # For pixel error calculation

                    with autocast(enabled=use_amp):
                        val_pred_offsets, val_pred_logits = model(val_img, val_in_psf)
                        batch_loss_comb, batch_cls_loss, batch_reg_loss = criterion(
                            val_pred_offsets, val_pred_logits, val_tgt_offsets, val_tgt_labels
                        )
                    
                    total_val_loss_combined += batch_loss_comb.item() * val_img.size(0)
                    total_val_cls_loss += batch_cls_loss.item() * val_img.size(0)
                    total_val_reg_loss += batch_reg_loss.item() * val_img.size(0)
                    total_val_samples += val_img.size(0)

                    # Calculate pixel error for validation
                    if MODEL_INPUT_SIZE > 1 and val_pred_offsets is not None and val_pred_logits is not None:
                        # Get final predicted coordinate from offsets and best anchor
                        probs_val = torch.sigmoid(val_pred_logits) # (B, N_anchors, 1)
                        best_anchor_indices_batch = torch.argmax(probs_val.squeeze(-1), dim=1) # (B,)
                        
                        # Gather offsets for the best anchors:
                        # Need to use gather or iterate if ANCHOR_POINTS_NORM_VAL is not directly batch-indexed
                        # Simplest: ANCHOR_POINTS_NORM_VAL is (N_anchors, 2). Expand for batch if needed.
                        # selected_pred_offsets = val_pred_offsets[torch.arange(val_img.size(0)), best_anchor_indices_batch, :]
                        
                        batch_pred_coords_norm_list = []
                        for b_idx in range(val_img.size(0)):
                            best_anchor_idx = best_anchor_indices_batch[b_idx]
                            selected_offset = val_pred_offsets[b_idx, best_anchor_idx, :]
                            selected_anchor_coord = ANCHOR_POINTS_NORM_VAL[best_anchor_idx, :] # ANCHOR_POINTS_NORM_VAL is on DEVICE
                            final_pred_coord_norm = selected_anchor_coord + selected_offset
                            batch_pred_coords_norm_list.append(final_pred_coord_norm)
                        
                        batch_pred_coords_norm = torch.stack(batch_pred_coords_norm_list) # (B, 2)

                        scale_factor = MODEL_INPUT_SIZE - 1
                        pred_px_x = batch_pred_coords_norm[:, 0] * scale_factor
                        pred_px_y = batch_pred_coords_norm[:, 1] * scale_factor
                        tgt_px_x = val_direct_tgt_coords[:, 0] * scale_factor # Use direct GT coords
                        tgt_px_y = val_direct_tgt_coords[:, 1] * scale_factor
                        
                        pixel_errors_batch = torch.sqrt(
                            (pred_px_x - tgt_px_x)**2 + (pred_px_y - tgt_px_y)**2
                        )
                        total_val_pixel_error += pixel_errors_batch.sum().item()

                    if not printed_val_coords_this_interval and val_img.size(0) > 0:
                        print(f"\n--- Validation Final Coordinate Comparison (Iter {iteration}, Val Batch 1) ---")
                        pred_final_norm = batch_pred_coords_norm[0].cpu().numpy()
                        target_final_norm = val_direct_tgt_coords[0].cpu().numpy()
                        print(f"  Sample 0 Pred Final: [{pred_final_norm[0]:.3f}, {pred_final_norm[1]:.3f}] --- "
                              f"Target Final: [{target_final_norm[0]:.3f}, {target_final_norm[1]:.3f}]")
                        print("-----------------------------------------------------------------------------------")
                        printed_val_coords_this_interval = True


            random.setstate(rng_state['random']); np.random.set_state(rng_state['numpy']); torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng_state['cuda'])
            set_seed(SEED + iteration + VALIDATION_BATCHES + 1) # Restore seeding for next training steps

            avg_val_loss_combined = total_val_loss_combined / total_val_samples if total_val_samples > 0 else float('inf')
            avg_val_cls_loss = total_val_cls_loss / total_val_samples if total_val_samples > 0 else float('inf')
            avg_val_reg_loss = total_val_reg_loss / total_val_samples if total_val_samples > 0 else float('inf')
            average_val_pixel_error = total_val_pixel_error / total_val_samples if total_val_samples > 0 else float('inf')
            
            iterations_list.append(iteration)
            train_loss_list.append(avg_train_loss) # Store combined train loss
            val_loss_list_combined.append(avg_val_loss_combined)
            val_pixel_error_list.append(average_val_pixel_error)


            log_message = (f"Iter [{iteration}/{TOTAL_ITERATIONS}] | Train Loss (Comb/Cls/Reg): {avg_train_loss:.4f}/{avg_train_cls_loss:.4f}/{avg_train_reg_loss:.4f} | "
                           f"Val Loss (Comb/Cls/Reg): {avg_val_loss_combined:.4f}/{avg_val_cls_loss:.4f}/{avg_val_reg_loss:.4f} | Val Pixel Err: {average_val_pixel_error:.2f}px | "
                           f"LR: {optimizer.param_groups[0]['lr']:.4e}")
            print(f"\n{log_message}")

            with open(LOG_FILE_PATH, "a") as log_file: log_file.write(log_message + "\n")
            
            # Save best model based on pixel error
            if average_val_pixel_error < best_val_pixel_error:
                best_val_pixel_error = average_val_pixel_error
                best_val_loss = avg_val_loss_combined # Store corresponding combined loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    -> New best model saved with Val Pixel Err: {best_val_pixel_error:.2f}px (Val Comb Loss: {best_val_loss:.4f})")
            
            train_loss_accum, train_cls_loss_accum, train_reg_loss_accum, samples_in_accum = 0.0, 0.0, 0.0, 0
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Iter {iteration}/{TOTAL_ITERATIONS} | Last Batch Loss (Total/Cls/Reg): {loss.item():.4f}/{cls_loss.item():.4f}/{reg_loss.item():.4f} | LR: {current_lr:.2e}")


    print("Training complete!")
    pbar.close()

    # Plotting (Combined Loss and Pixel Error)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Combined Loss', color=color)
    ax1.plot(iterations_list, train_loss_list, label='Train Combined Loss', color=color, linestyle='--')
    ax1.plot(iterations_list, val_loss_list_combined, label='Validation Combined Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Validation Pixel Error (px)', color=color)
    ax2.plot(iterations_list, val_pixel_error_list, label='Validation Pixel Error', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("Training & Validation Metrics (Offset Regression) over Iterations")
    fig.tight_layout() # otherwise the right y-label is slightly clipped
    plot_path = os.path.join(OUTPUT_DIR, "training_metrics_plot_offsetreg.png")
    plt.savefig(plot_path); plt.close()


    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (Val Pixel Err: {best_val_pixel_error:.4f}px, Val Comb Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
