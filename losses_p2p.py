#losses_p2p.py
"""
Loss functions for model training (Offset Regression Hybrid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config_p2p import (
    REGRESSION_LOSS_TYPE, CLASSIFICATION_LOSS_TYPE,
    LOSS_CLS_WEIGHT, LOSS_REG_WEIGHT
)

# TODO: If using focal loss, implement it here.
# class FocalLoss(nn.Module): ...


def combined_offset_regression_loss(
    predicted_offsets, predicted_logits, 
    target_offsets, target_labels
    ):
    """
    Computes combined loss for offset regression and classification.

    Args:
        predicted_offsets (torch.Tensor): (B, N_anchors, 2)
        predicted_logits (torch.Tensor): (B, N_anchors, 1) or (B, N_anchors, num_classes)
        target_offsets (torch.Tensor): (B, N_anchors, 2)
        target_labels (torch.Tensor): (B, N_anchors, 1) - should be 0.0 or 1.0 for BCE

    Returns:
        tuple: (total_loss, classification_loss, regression_loss)
    """
    B, N_anchors, _ = predicted_offsets.shape

    # --- Classification Loss ---
    # predicted_logits are raw logits from the model head
    # target_labels are 0.0 or 1.0
    if CLASSIFICATION_LOSS_TYPE == 'bce':
        # Squeeze last dim of logits if it's 1 (binary case)
        cls_logits_for_loss = predicted_logits.squeeze(-1) if predicted_logits.shape[-1] == 1 else predicted_logits
        cls_targets_for_loss = target_labels.squeeze(-1) if target_labels.shape[-1] == 1 else target_labels
        
        cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean') # Averages over all B*N_anchors predictions
        classification_loss = cls_loss_fn(cls_logits_for_loss, cls_targets_for_loss)
    # elif CLASSIFICATION_LOSS_TYPE == 'focal':
        # focal_loss_fn = FocalLoss(...) # Ensure it handles logits
        # classification_loss = focal_loss_fn(predicted_logits, target_labels)
    else:
        raise ValueError(f"Unsupported classification loss type: {CLASSIFICATION_LOSS_TYPE}")

    # --- Regression Loss (only for positive anchors) ---
    # Mask for positive anchors (where target_label is 1.0)
    positive_mask = (target_labels > 0.5).squeeze(-1) # (B, N_anchors)
    num_positives = positive_mask.sum().float().clamp(min=1.0) # Total positives in batch

    # Select only predictions and targets for positive anchors
    # Expand positive_mask to be (B, N_anchors, 2) to match offset shapes
    positive_mask_for_offsets = positive_mask.unsqueeze(-1).expand_as(predicted_offsets)
    
    # These will be flattened tensors of shape (Num_positives_in_batch, 2)
    pred_offsets_positive = predicted_offsets[positive_mask_for_offsets].view(-1, 2)
    target_offsets_positive = target_offsets[positive_mask_for_offsets].view(-1, 2)

    if pred_offsets_positive.numel() > 0: # If there are any positive anchors in the batch
        if REGRESSION_LOSS_TYPE == 'smooth_l1':
            reg_loss_fn = nn.SmoothL1Loss(reduction='sum')
        elif REGRESSION_LOSS_TYPE == 'mse':
            reg_loss_fn = nn.MSELoss(reduction='sum')
        else:
            raise ValueError(f"Unsupported regression loss type: {REGRESSION_LOSS_TYPE}")
        
        regression_loss = reg_loss_fn(pred_offsets_positive, target_offsets_positive) / num_positives
    else:
        # No positive anchors in the batch, so regression loss is 0
        regression_loss = torch.tensor(0.0, device=predicted_offsets.device, dtype=predicted_offsets.dtype)
    
    # Combine losses
    total_loss = (LOSS_CLS_WEIGHT * classification_loss) + (LOSS_REG_WEIGHT * regression_loss)
    
    return total_loss, classification_loss, regression_loss