#model_p2p.py
"""
Model definitions including VGG19 Encoder, FPN Decoder, ASPP, and Offset Regression Heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Import from config
from config_p2p import (
    MODEL_INPUT_SIZE, HEAD_NUM_ANCHORS_PER_LOC, HEAD_NUM_CLASSES, 
    HEAD_FPN_OUTPUT_STRIDE
)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]): # Default rates from DeepLab
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        for rate in rates[1:]: # Start from rates[1] because rate=0 is 1x1 conv
             self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                        padding=rate, dilation=rate, bias=False))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.bn_ops = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(len(self.convs) + 1)])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(self.convs) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2) 
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []
        for i, conv in enumerate(self.convs):
            features.append(F.relu(self.bn_ops[i](conv(x))))
        gap_feat = self.global_pool(x)
        gap_feat = F.interpolate(gap_feat, size=size, mode='bilinear', align_corners=False)
        features.append(self.bn_ops[-1](gap_feat)) # Note: P2PNet has self.bn_ops[-1](gap_feat)
        x = torch.cat(features, dim=1)
        x = self.project(x)
        return x

class VGG19Encoder(nn.Module):
    def __init__(self):
        super(VGG19Encoder, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        features = list(vgg19.features)
        self.feature_layers = nn.ModuleList(features)
        # VGG19 feature indices (after ReLU, before MaxPool):
        # C1: features[3] (stride 1, 224x224 for 224 input)
        # C2: features[8] (stride 2, 112x112)
        # C3: features[17] (stride 4, 56x56)
        # C4: features[26] (stride 8, 28x28)
        # C5: features[35] (stride 16, 14x14)
        self.capture_indices = {3: 'C1', 8: 'C2', 17: 'C3', 26: 'C4', 35: 'C5'} 

    def forward(self, x):
        results = {}
        for i, layer in enumerate(self.feature_layers):
            x = layer(x)
            if i in self.capture_indices:
                 results[self.capture_indices[i]] = x
        return [results['C1'], results['C2'], results['C3'], results['C4'], results['C5']]

class SmallPSFEncoder(nn.Module):
    def __init__(self):
        super(SmallPSFEncoder, self).__init__()
        # Input PSF is (MODEL_INPUT_SIZE / HEAD_FPN_OUTPUT_STRIDE)
        # e.g. 224 / 8 = 28x28 for mask features to match C5.
        # Or, it can be full res and downsampled here.
        # P2PNet mask_encoder outputs 14x14 for 224 input (stride 16), to match C5.
        # Let's assume input_psf for this encoder is MODEL_INPUT_SIZE, and it downsamples to match C5.
        self.encoder = nn.Sequential( # Output H/16, W/16
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), # H/2
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), # H/4
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), # H/8
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), # H/16
            nn.Conv2d(32, 64, kernel_size=1), nn.ReLU(inplace=True) # Keep 64 channels
        )
    def forward(self, x): return self.encoder(x)

class FPNDecoder(nn.Module):
    # encoder_channels: [C1_ch, C2_ch, C3_ch, C4_ch, C5_fused_ch]
    # fpn_channels: internal channel dimension for FPN layers
    # out_channels_final_conv: output channels of the conv applied to the selected FPN level
    def __init__(self, encoder_channels=[64, 128, 256, 512, 512], fpn_channels=256, out_channels_final_conv=64):
        super(FPNDecoder, self).__init__()
        assert len(encoder_channels) == 5
        self.lateral_convs = nn.ModuleList()
        # C5_fused, C4, C3, C2, C1
        for enc_ch in reversed(encoder_channels): # From C5_fused down to C1
            self.lateral_convs.append(nn.Conv2d(enc_ch, fpn_channels, kernel_size=1))
        
        self.smooth_convs = nn.ModuleList()
        for _ in range(len(encoder_channels)): # P5, P4, P3, P2, P1
             self.smooth_convs.append(nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1))
        
        # This conv will be applied to the chosen FPN output level (e.g., P4-equiv)
        self.final_conv_on_selected_level = nn.Conv2d(fpn_channels, out_channels_final_conv, kernel_size=3, padding=1)

    def _upsample_add(self, top_down_feat, lateral_feat):
        _, _, H, W = lateral_feat.shape
        upsampled_feat = F.interpolate(top_down_feat, size=(H, W), mode='bilinear', align_corners=False)
        return upsampled_feat + lateral_feat

    def forward(self, x_top_fused_c5, encoder_features_c1_c4): # x_top_fused_c5 is after ASPP
        C1, C2, C3, C4 = encoder_features_c1_c4
        all_encoder_levels_for_fpn = [C1, C2, C3, C4, x_top_fused_c5] # Order: C1 to C5_fused
        
        pyramid_features = [] # Will store P5_smooth, P4_smooth, ..., P1_smooth

        # P5 (from C5_fused)
        p = self.lateral_convs[0](all_encoder_levels_for_fpn[-1]) # Lateral from C5_fused
        p = self.smooth_convs[0](p) # P5_smooth
        pyramid_features.append(p)

        # P4 down to P1
        for i in range(1, len(self.lateral_convs)):
            lateral_input_idx = len(all_encoder_levels_for_fpn) - 1 - i # C4, then C3, ...
            lateral_feat = self.lateral_convs[i](all_encoder_levels_for_fpn[lateral_input_idx])
            
            p_prev_upsampled = F.interpolate(pyramid_features[-1], size=lateral_feat.shape[2:], mode='bilinear', align_corners=False)
            top_down_feat = p_prev_upsampled + lateral_feat
            
            p = self.smooth_convs[i](top_down_feat) # Pi_smooth
            pyramid_features.append(p)
        
        # pyramid_features is now [P5_smooth, P4_smooth, P3_smooth, P2_smooth, P1_smooth]
        # P5_smooth has stride 16 (from C5)
        # P4_smooth has stride 8 (from C4)
        # P3_smooth has stride 4 (from C3)
        # P2_smooth has stride 2 (from C2)
        # P1_smooth has stride 1 (from C1)

        # Select the FPN level corresponding to HEAD_FPN_OUTPUT_STRIDE
        if HEAD_FPN_OUTPUT_STRIDE == 16: selected_fpn_level_feat = pyramid_features[0] # P5_smooth
        elif HEAD_FPN_OUTPUT_STRIDE == 8: selected_fpn_level_feat = pyramid_features[1] # P4_smooth
        elif HEAD_FPN_OUTPUT_STRIDE == 4: selected_fpn_level_feat = pyramid_features[2] # P3_smooth
        elif HEAD_FPN_OUTPUT_STRIDE == 2: selected_fpn_level_feat = pyramid_features[3] # P2_smooth
        elif HEAD_FPN_OUTPUT_STRIDE == 1: selected_fpn_level_feat = pyramid_features[4] # P1_smooth
        else:
            raise ValueError(f"Unsupported HEAD_FPN_OUTPUT_STRIDE: {HEAD_FPN_OUTPUT_STRIDE}")

        out = F.relu(self.final_conv_on_selected_level(selected_fpn_level_feat))
        return out


# Adapted from P2PNet's RegressionModel and ClassificationModel
class OffsetRegressionHead(nn.Module):
    def __init__(self, in_channels, num_anchors_per_loc=1, feature_size=256):
        super(OffsetRegressionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        # Output 2 values (dx, dy) for each anchor at each spatial location
        self.output_conv = nn.Conv2d(feature_size, num_anchors_per_loc * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output_conv(out) # (B, num_anchors_per_loc * 2, H_feat, W_feat)
        # No final activation for offsets (can be positive or negative)
        return out

class LogitClassificationHead(nn.Module):
    def __init__(self, in_channels, num_anchors_per_loc=1, num_classes=1, feature_size=256):
        super(LogitClassificationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        # Output logits for each class for each anchor at each spatial location
        self.output_conv = nn.Conv2d(feature_size, num_anchors_per_loc * num_classes, kernel_size=3, padding=1)
        # No final sigmoid here if using BCEWithLogitsLoss

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output_conv(out) # (B, num_anchors_per_loc * num_classes, H_feat, W_feat)
        return out


class VGG19FPNASPP(nn.Module):
    def __init__(self):
        super(VGG19FPNASPP, self).__init__()
        self.image_encoder = VGG19Encoder()
        self.mask_encoder = SmallPSFEncoder() # Expects full-res PSF, outputs features at C5 res

        vgg_c1_ch, vgg_c2_ch, vgg_c3_ch, vgg_c4_ch, vgg_c5_ch = 64, 128, 256, 512, 512
        mask_feat_ch = 64 # Output of SmallPSFEncoder
        
        # Fusion happens at C5 level (stride 16)
        fusion_in_ch_c5 = vgg_c5_ch + mask_feat_ch
        fusion_out_ch_c5 = 512 # Channel dim for fused C5 features before ASPP

        self.fusion_conv_c5 = nn.Sequential(
            nn.Conv2d(fusion_in_ch_c5, fusion_out_ch_c5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_ch_c5), nn.ReLU(inplace=True)
        )
        self.aspp_c5 = ASPP(in_channels=fusion_out_ch_c5, out_channels=fusion_out_ch_c5)
        
        fpn_encoder_channels = [vgg_c1_ch, vgg_c2_ch, vgg_c3_ch, vgg_c4_ch, fusion_out_ch_c5]
        fpn_internal_channels = 256
        fpn_final_conv_out_channels = 64 # Output channels of FPN level selected for heads

        self.fpn_decoder = FPNDecoder(
             encoder_channels=fpn_encoder_channels,
             fpn_channels=fpn_internal_channels, 
             out_channels_final_conv=fpn_final_conv_out_channels
        )
        
        self.regression_head = OffsetRegressionHead(
            in_channels=fpn_final_conv_out_channels, 
            num_anchors_per_loc=HEAD_NUM_ANCHORS_PER_LOC,
            feature_size=256 # Internal feature size of the head
        )
        self.classification_head = LogitClassificationHead(
            in_channels=fpn_final_conv_out_channels,
            num_anchors_per_loc=HEAD_NUM_ANCHORS_PER_LOC,
            num_classes=HEAD_NUM_CLASSES, # Should be 1 for "is_target_point"
            feature_size=256 # Internal feature size of the head
        )
        self.head_num_classes = HEAD_NUM_CLASSES


    def forward(self, image, mask):
        if mask.dim() == 3: mask = mask.unsqueeze(1) # Ensure (B,1,H,W)

        C1, C2, C3, C4, C5 = self.image_encoder(image)
        mask_features_c5_res = self.mask_encoder(mask) # (B, 64, H_img/16, W_img/16)

        # Ensure mask_features are spatially compatible with C5 for concatenation
        if C5.shape[2:] != mask_features_c5_res.shape[2:]:
            mask_features_c5_res = F.interpolate(mask_features_c5_res, size=C5.shape[2:], mode='bilinear', align_corners=False)

        fused_features_c5 = torch.cat([C5, mask_features_c5_res], dim=1)
        fused_c5_processed = self.fusion_conv_c5(fused_features_c5)
        aspp_output_c5 = self.aspp_c5(fused_c5_processed) # (B, fusion_out_ch_c5, H_c5, W_c5)
        
        # FPN decoder will select the appropriate level based on HEAD_FPN_OUTPUT_STRIDE
        # The output `fpn_selected_level_output` will have HEAD_FPN_OUTPUT_STRIDE
        # e.g., for stride 8, it's (B, fpn_final_conv_out_channels, H_img/8, W_img/8)
        fpn_selected_level_output = self.fpn_decoder(aspp_output_c5, [C1, C2, C3, C4])
        
        pred_offsets_raw = self.regression_head(fpn_selected_level_output)
        pred_logits_raw = self.classification_head(fpn_selected_level_output)
        
        # Reshape/permute from (B, A*channels_per_anchor, H_feat, W_feat)
        # to (B, H_feat*W_feat*A, channels_per_anchor)
        B, _, H_feat, W_feat = pred_offsets_raw.shape # A*2 channels
        
        # (B, H_feat, W_feat, A*2) -> (B, H_feat*W_feat*A, 2)
        pred_offsets = pred_offsets_raw.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        
        # (B, H_feat, W_feat, A*num_classes) -> (B, H_feat*W_feat*A, num_classes)
        pred_logits = pred_logits_raw.permute(0, 2, 3, 1).contiguous().view(B, -1, self.head_num_classes)
        
        return pred_offsets, pred_logits
