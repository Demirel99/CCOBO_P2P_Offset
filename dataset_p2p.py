# dataset_p2p.py
"""
Dataset preparation, augmentation, and sample generation functions.
"""
import numpy as np
import cv2
import random
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import os
import glob 
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle

# Import from config
from config_p2p import (
    AUGMENTATION_SIZE, MODEL_INPUT_SIZE, MIN_DIM_RESCALE, GT_PSF_SIGMA,
    IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, DEVICE, HEAD_FPN_OUTPUT_STRIDE,
    HEAD_NUM_ANCHORS_PER_LOC
)
from utils import generate_anchor_grid_centers # Import the new utility


# --- ImageNet Mean/Std for Normalization ---
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Pre-generate anchor coordinates (normalized) ---
# These are fixed for a given model input size and FPN stride
# Assuming HEAD_NUM_ANCHORS_PER_LOC is 1 for simplicity (one anchor center per grid cell)
FEATURE_MAP_H = MODEL_INPUT_SIZE // HEAD_FPN_OUTPUT_STRIDE
FEATURE_MAP_W = MODEL_INPUT_SIZE // HEAD_FPN_OUTPUT_STRIDE
# ANCHOR_POINTS_NORM shape: ( (H_feat * W_feat * HEAD_NUM_ANCHORS_PER_LOC), 2)
# For HEAD_NUM_ANCHORS_PER_LOC = 1, it's ( (H_feat * W_feat), 2)
ANCHOR_POINTS_NORM = generate_anchor_grid_centers(FEATURE_MAP_H, FEATURE_MAP_W)
NUM_TOTAL_ANCHORS = ANCHOR_POINTS_NORM.shape[0]


def prepare_data_augmentations(image, gt_coor, target_size=AUGMENTATION_SIZE, min_dim=MIN_DIM_RESCALE):
    """
    Applies random scaling, cropping, and flipping augmentations.
    Outputs an image and corresponding coordinates at the target_size.

    Args:
        image (np.ndarray): Input image (H, W, C).
        gt_coor (np.ndarray): Ground truth coordinates (N, 2) as (x, y).
        target_size (int): The size of the square crop to extract.
        min_dim (int): Minimum dimension allowed after scaling.

    Returns:
        tuple: (augmented_image, augmented_gt_coor)
               augmented_image is (target_size, target_size, C)
               augmented_gt_coor is (M, 2) for points within the final crop.
    """
    if image is None:
        print("Warning: prepare_data_augmentations received None image.")
        return None, None
    h, w, c = image.shape
    if c != 3:
        # Try to convert if grayscale, or if alpha channel exists
        if c == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif c == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            print(f"Warning: Expected a 3-channel image (H, W, 3), but got {c} channels. Skipping.")
            return None, None
        h, w, c = image.shape # Re-check shape

    # 1. Random Scale
    scale_factor = random.uniform(0.7, 1.3)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if min(h, w) > 0 :
        if min(new_h, new_w) < min_dim:
            scale_factor = min_dim / min(h, w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    elif min(h,w) == 0:
        print("Warning: Image has zero dimension before scaling.")
        return None, None 


    new_h = max(1, new_h)
    new_w = max(1, new_w)

    try:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Error during cv2.resize: {e}. Original shape: {(h,w)}, Target shape: {(new_h, new_w)}")
        return None, None

    scaled_gt_coor = gt_coor * scale_factor if gt_coor is not None and gt_coor.size > 0 else np.array([]) 

    # 2. Random Crop (to target_size)
    curr_h, curr_w = scaled_image.shape[:2]
    crop_h = min(target_size, curr_h)
    crop_w = min(target_size, curr_w)

    cropped_gt_coor = np.array([]) 

    if curr_h >= target_size and curr_w >= target_size:
        x_min = random.randint(0, curr_w - target_size)
        y_min = random.randint(0, curr_h - target_size)
        cropped_image = scaled_image[y_min : y_min + target_size, x_min : x_min + target_size]
        
        if scaled_gt_coor.size > 0:
            keep_mask = (scaled_gt_coor[:, 0] >= x_min) & (scaled_gt_coor[:, 0] < x_min + target_size) & \
                        (scaled_gt_coor[:, 1] >= y_min) & (scaled_gt_coor[:, 1] < y_min + target_size)
            cropped_gt_coor = scaled_gt_coor[keep_mask]
            if cropped_gt_coor.size > 0: 
                cropped_gt_coor[:, 0] -= x_min
                cropped_gt_coor[:, 1] -= y_min
        final_h, final_w = target_size, target_size
    else:
        print(f"Warning: Scaled image ({curr_h}x{curr_w}) smaller than target ({target_size}x{target_size}). Resizing up (may distort aspect ratio).")
        try:
             cropped_image = cv2.resize(scaled_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             print(f"Error during cv2.resize (upscaling): {e}. Scaled shape: {(curr_h, curr_w)}, Target shape: {(target_size, target_size)}")
             return None, None

        if curr_w > 0 and curr_h > 0 and scaled_gt_coor.size > 0:
            scale_x_final = target_size / float(curr_w)
            scale_y_final = target_size / float(curr_h)
            cropped_gt_coor = scaled_gt_coor.copy()
            cropped_gt_coor[:, 0] *= scale_x_final
            cropped_gt_coor[:, 1] *= scale_y_final
            cropped_gt_coor[:, 0] = np.clip(cropped_gt_coor[:, 0], 0, target_size - 1)
            cropped_gt_coor[:, 1] = np.clip(cropped_gt_coor[:, 1], 0, target_size - 1)
        final_h, final_w = target_size, target_size

    # 3. Random Horizontal Flip
    if random.random() < 0.5:
        cropped_image = cv2.flip(cropped_image, 1) 
        if cropped_gt_coor.size > 0:
            cropped_gt_coor[:, 0] = (final_w - 1) - cropped_gt_coor[:, 0]

    if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
         print(f"Warning: Final augmented image shape {cropped_image.shape} does not match target {target_size}x{target_size}. Resizing again.")
         try:
            cropped_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
         except cv2.error as e:
            print(f"Error during final cv2.resize: {e}. Shape before: {cropped_image.shape}, Target shape: {(target_size, target_size)}")
            return None, None

    return cropped_image, cropped_gt_coor


def generate_single_psf(coord, image_shape, sigma):
    """Generates a single normalized Gaussian PSF centered at coord."""
    height, width = image_shape
    psf = np.zeros((height, width), dtype=np.float32)
    x = np.clip(int(round(coord[0])), 0, width - 1)
    y = np.clip(int(round(coord[1])), 0, height - 1)
    psf[y, x] = 1.0
    psf = gaussian_filter(psf, sigma=sigma, order=0, mode='constant', cval=0.0)
    psf_sum = np.sum(psf)
    if psf_sum > 1e-7:
        psf /= psf_sum
    return psf

def get_center_crop_coords(image_size, crop_size):
    """Calculates the top-left (y, x) coordinates for a center crop."""
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    start_y = max(0, (img_h - crop_h) // 2)
    start_x = max(0, (img_w - crop_w) // 2)
    return start_y, start_x

def get_coords_in_center_crop(coordinates, image_shape, center_crop_shape):
    """Returns indices of coordinates falling within the central crop area."""
    if coordinates is None or coordinates.shape[0] == 0: return []
    img_h, img_w = image_shape
    crop_h, crop_w = center_crop_shape
    start_y, start_x = get_center_crop_coords(image_shape, center_crop_shape)
    end_y, end_x = start_y + crop_h, start_x + crop_w

    indices_in_center = [
        i for i, (x, y) in enumerate(coordinates)
        if start_x <= x < end_x and start_y <= y < end_y
    ]
    return indices_in_center


def generate_train_sample(image_paths, gt_paths, augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA):
    """
    Generates a single training sample for offset regression.
    Returns: image, input_psf, target_offsets, target_labels, direct_target_coord_for_eval
    """
    max_retries = 10
    for _ in range(max_retries):
        rand_idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[rand_idx]
        img_filename = os.path.basename(image_path)
        
        gt_filename_expected = "GT_" + os.path.splitext(img_filename)[0] + ".mat"
        gt_path_base = os.path.dirname(gt_paths[0]) if gt_paths else os.path.dirname(image_path).replace("images", "ground_truth")
        gt_path = os.path.join(gt_path_base, gt_filename_expected)

        if not os.path.exists(gt_path):
             if rand_idx < len(gt_paths) and os.path.exists(gt_paths[rand_idx]):
                 gt_path = gt_paths[rand_idx]
             else:
                 print(f"Warning: GT file for {image_path} not found. Skipping.")
                 continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping.")
            continue
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3:
             print(f"Warning: Image {image_path} has unexpected shape {image.shape}. Skipping.")
             continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            mat_data = loadmat(gt_path)
            if 'image_info' in mat_data:
                 gt_coor = mat_data['image_info'][0, 0][0, 0][0].astype(np.float32)
            elif 'annPoints' in mat_data:
                 gt_coor = mat_data['annPoints'].astype(np.float32)
            else: 
                 found_coords = False
                 for key, value in mat_data.items():
                     if isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[1] == 2:
                         gt_coor = value.astype(np.float32); found_coords = True; break
                 if not found_coords: 
                     print(f"Warning: No coordinate data in {gt_path}. Skipping.")
                     continue
        except Exception as e:
             print(f"Warning: Error loading/parsing .mat {gt_path}: {e}. Skipping.")
             continue

        if gt_coor.shape[0] == 0: continue

        aug_image, aug_gt_coor = prepare_data_augmentations(image, gt_coor.copy(), target_size=augment_size)
        if aug_image is None or aug_gt_coor is None or aug_gt_coor.shape[0] < 1: continue
        
        img_h, img_w = aug_image.shape[:2]
        if img_h == 0 or img_w == 0: continue

        sorted_indices = np.lexsort((aug_gt_coor[:, 0], -aug_gt_coor[:, 1]))
        sorted_aug_gt_coor = aug_gt_coor[sorted_indices]

        center_crop_shape = (model_input_size, model_input_size)
        indices_of_sorted_coords_in_center = get_coords_in_center_crop(
            sorted_aug_gt_coor, (img_h, img_w), center_crop_shape
        )
        if not indices_of_sorted_coords_in_center: continue
        timestep = random.choice(indices_of_sorted_coords_in_center)
        
        input_psf_full = np.zeros((img_h, img_w), dtype=np.float32)
        if timestep > 0:
            for i in range(timestep):
                input_psf_full += generate_single_psf(sorted_aug_gt_coor[i], (img_h, img_w), psf_sigma)
        
        actual_target_coord_in_aug = sorted_aug_gt_coor[timestep]

        start_y, start_x = get_center_crop_coords((img_h, img_w), center_crop_shape)
        end_y, end_x = start_y + model_input_size, start_x + model_input_size

        if start_y < 0 or start_x < 0 or end_y > img_h or end_x > img_w:
            print(f"Warning: Invalid crop coordinates for {image_path}. Skipping.")
            continue

        final_image = aug_image[start_y:end_y, start_x:end_x]
        final_input_psf = input_psf_full[start_y:end_y, start_x:end_x]

        target_coord_cropped_x = actual_target_coord_in_aug[0] - start_x
        target_coord_cropped_y = actual_target_coord_in_aug[1] - start_y

        if not (0 <= target_coord_cropped_x < model_input_size and \
                0 <= target_coord_cropped_y < model_input_size):
            print(f"Warning: Target coordinate outside final crop for {image_path}. Skipping.")
            continue
            
        norm_target_x = target_coord_cropped_x / (model_input_size -1) if model_input_size > 1 else 0.0
        norm_target_y = target_coord_cropped_y / (model_input_size -1) if model_input_size > 1 else 0.0
        direct_target_coord_normalized = np.array([norm_target_x, norm_target_y], dtype=np.float32)

        # --- Generate targets for offset regression ---
        gt_coord_norm_tensor = torch.from_numpy(direct_target_coord_normalized).unsqueeze(0) # (1, 2)
        
        # ANCHOR_POINTS_NORM is (N_total_anchors, 2)
        distances_sq = torch.sum((ANCHOR_POINTS_NORM.unsqueeze(0) - gt_coord_norm_tensor)**2, dim=2) # (1, N_total_anchors)
        positive_anchor_idx = torch.argmin(distances_sq.squeeze(0)).item() # Squeeze to (N_total_anchors)
        
        target_offsets = torch.zeros(NUM_TOTAL_ANCHORS, 2, dtype=torch.float32)
        target_labels = torch.zeros(NUM_TOTAL_ANCHORS, 1, dtype=torch.float32) # For BCEWithLogitsLoss
        
        positive_anchor_coords = ANCHOR_POINTS_NORM[positive_anchor_idx] # (2,)
        offset_calc_x = direct_target_coord_normalized[0] - positive_anchor_coords[0].item()
        offset_calc_y = direct_target_coord_normalized[1] - positive_anchor_coords[1].item()
        
        target_offsets[positive_anchor_idx, :] = torch.tensor([offset_calc_x, offset_calc_y])
        target_labels[positive_anchor_idx, 0] = 1.0
        # --- End target generation for offset regression ---

        if final_image.shape[:2] != center_crop_shape or final_input_psf.shape != center_crop_shape:
            print(f"Warning: Cropped shape mismatch for {image_path}. Skipping.")
            continue

        final_image_tensor = torch.from_numpy(final_image.copy()).permute(2, 0, 1).float() / 255.0
        final_image_tensor = (final_image_tensor - IMG_MEAN) / IMG_STD

        max_val = np.max(final_input_psf)
        if max_val > 1e-7: final_input_psf = final_input_psf / max_val
        final_input_psf_tensor = torch.from_numpy(final_input_psf).float().unsqueeze(0)

        direct_target_coord_tensor_for_eval = torch.from_numpy(direct_target_coord_normalized).float()

        expected_img_shape = (3, model_input_size, model_input_size)
        expected_psf_shape = (1, model_input_size, model_input_size)
        if final_image_tensor.shape != expected_img_shape or \
           final_input_psf_tensor.shape != expected_psf_shape or \
           target_offsets.shape != (NUM_TOTAL_ANCHORS, 2) or \
           target_labels.shape != (NUM_TOTAL_ANCHORS, 1) or \
           direct_target_coord_tensor_for_eval.shape != (2,):
            print(f"Warning: Final shape mismatch for {image_path}. Retrying. Img: {final_image_tensor.shape}, PSF: {final_input_psf_tensor.shape}, TgtOff: {target_offsets.shape}, TgtLab: {target_labels.shape}")
            continue
            
        return final_image_tensor, final_input_psf_tensor, target_offsets, target_labels, direct_target_coord_tensor_for_eval

    print(f"Warning: Failed to generate a valid sample after {max_retries} retries. Returning None tuple.")
    return None, None, None, None, None


def generate_batch(image_paths, gt_paths, batch_size, generation_fn=generate_train_sample, **kwargs):
    """Generates a batch of data using the specified generation function."""
    image_batch, input_psf_batch, target_offsets_batch, target_labels_batch, direct_coords_eval_batch = [], [], [], [], []
    attempts = 0
    max_attempts = batch_size * 10 

    while len(image_batch) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            sample = generation_fn(image_paths, gt_paths, **kwargs)

            if sample is not None and sample[0] is not None:
                img, in_psf, tgt_offsets, tgt_labels, direct_coords_eval = sample
                if all(isinstance(t, torch.Tensor) for t in sample):
                    image_batch.append(img)
                    input_psf_batch.append(in_psf)
                    target_offsets_batch.append(tgt_offsets)
                    target_labels_batch.append(tgt_labels)
                    direct_coords_eval_batch.append(direct_coords_eval)
                else:
                    print(f"Warning: generation_fn returned non-Tensor data. Skipping.")
        except Exception as e:
            import traceback
            print(f"Error during sample generation: {e}. Skipping sample.")
            print(traceback.format_exc())
            continue

    if not image_batch:
        print(f"Warning: Failed to generate any valid samples for a batch after {max_attempts} attempts.")
        return None, None, None, None, None

    try:
        final_image_batch = torch.stack(image_batch)
        final_input_psf_batch = torch.stack(input_psf_batch)
        final_target_offsets_batch = torch.stack(target_offsets_batch)
        final_target_labels_batch = torch.stack(target_labels_batch)
        final_direct_coords_eval_batch = torch.stack(direct_coords_eval_batch)
        return final_image_batch, final_input_psf_batch, final_target_offsets_batch, final_target_labels_batch, final_direct_coords_eval_batch
    except Exception as e:
        print(f"Error during torch.stack: {e}")
        if image_batch: print("Individual image shapes:", [t.shape for t in image_batch])
        if input_psf_batch: print("Individual input PSF shapes:", [t.shape for t in input_psf_batch])
        if target_offsets_batch: print("Individual target offset shapes:", [t.shape for t in target_offsets_batch])
        if target_labels_batch: print("Individual target label shapes:", [t.shape for t in target_labels_batch])
        if direct_coords_eval_batch: print("Individual direct coord shapes:", [t.shape for t in direct_coords_eval_batch])
        return None, None, None, None, None


# ==============================================================================
# Test Code - Visualize a generated sample (needs update for new targets)
# ==============================================================================
if __name__ == "__main__":
    print("Running dataset.py test (Offset Regression Mode)...")

    test_image_dir = IMAGE_DIR_TRAIN_VAL
    test_gt_dir = GT_DIR_TRAIN_VAL
    num_samples_to_show = 1 # Reduced for brevity

    print(f"Using Image Dir: {test_image_dir}")
    print(f"Using GT Dir: {test_gt_dir}")
    print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
    print(f"Anchor grid: {FEATURE_MAP_H}x{FEATURE_MAP_W}, Total anchors: {NUM_TOTAL_ANCHORS}")


    try:
        image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
        gt_paths = sorted(glob.glob(os.path.join(test_gt_dir, '*.mat')))
        if not image_paths: print(f"Error: No images found in {test_image_dir}"); exit()
    except Exception as e: print(f"Error finding dataset files: {e}"); exit()

    print(f"Found {len(image_paths)} images.")

    for i in range(num_samples_to_show):
        print(f"\n--- Generating Sample {i+1}/{num_samples_to_show} ---")
        sample_data = generate_train_sample(
            image_paths, gt_paths,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA
        )

        if sample_data is None or sample_data[0] is None:
            print("Failed to generate a valid sample. Skipping visualization.")
            continue

        img_tensor, input_psf_tensor, target_offsets, target_labels, direct_target_coord_eval = sample_data

        print(f"Generated Tensor Shapes:")
        print(f"  Image:                {img_tensor.shape}")
        print(f"  Input PSF:            {input_psf_tensor.shape}")
        print(f"  Target Offsets:       {target_offsets.shape}")
        print(f"  Target Labels:        {target_labels.shape}")
        print(f"  Direct Target Coords: {direct_target_coord_eval.shape}, Value: {direct_target_coord_eval.numpy()}")
        
        positive_anchor_idx = torch.argmax(target_labels.squeeze()).item()
        print(f"  Positive Anchor Index: {positive_anchor_idx}")
        print(f"  Positive Anchor Coords (norm): {ANCHOR_POINTS_NORM[positive_anchor_idx].numpy()}")
        print(f"  Offset for Positive Anchor: {target_offsets[positive_anchor_idx].numpy()}")


        img_vis = img_tensor.cpu() * IMG_STD.cpu() + IMG_MEAN.cpu()
        img_vis = torch.clamp(img_vis, 0, 1)
        img_vis_np = img_vis.permute(1, 2, 0).numpy()

        input_psf_np = input_psf_tensor.squeeze().cpu().numpy()
        
        direct_target_pixel_x = direct_target_coord_eval.numpy()[0] * (MODEL_INPUT_SIZE - 1)
        direct_target_pixel_y = direct_target_coord_eval.numpy()[1] * (MODEL_INPUT_SIZE - 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Generated Sample {i+1} (Offset Regression)', fontsize=16)

        axes[0].imshow(img_vis_np)
        axes[0].set_title(f'Input Image ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})')
        
        # Plot all anchor points as small blue dots
        for anchor_idx in range(NUM_TOTAL_ANCHORS):
            ax_norm, ay_norm = ANCHOR_POINTS_NORM[anchor_idx].numpy()
            ax_pixel = ax_norm * (MODEL_INPUT_SIZE -1)
            ay_pixel = ay_norm * (MODEL_INPUT_SIZE -1)
            color = 'lime' if anchor_idx == positive_anchor_idx else 'blue'
            radius = 3 if anchor_idx == positive_anchor_idx else 1
            alpha = 0.8 if anchor_idx == positive_anchor_idx else 0.3
            anchor_c = Circle((ax_pixel, ay_pixel), radius=radius, color=color, alpha=alpha)
            axes[0].add_patch(anchor_c)

        # Plot direct target coordinate as a red circle
        target_circle = Circle((direct_target_pixel_x, direct_target_pixel_y), radius=5, color='red', fill=False, linewidth=2)
        axes[0].add_patch(target_circle)
        axes[0].text(direct_target_pixel_x + 7, direct_target_pixel_y + 7, 
                     f'Actual Target\n({direct_target_pixel_x:.1f}, {direct_target_pixel_y:.1f})', 
                     color='red', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=1))
        axes[0].axis('off')

        im_in = axes[1].imshow(input_psf_np, cmap='viridis', vmin=0)
        axes[1].set_title(f'Input PSF (Sum of previous points)\nMax val: {np.max(input_psf_np):.4f}')
        axes[1].axis('off')
        fig.colorbar(im_in, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    print("\nDataset test finished.")