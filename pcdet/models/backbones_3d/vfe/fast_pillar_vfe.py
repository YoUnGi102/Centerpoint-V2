import torch
import torch.nn as nn
import torch.nn.functional as F
from .vfe_template import VFETemplate

class MAPELayer(nn.Module):
    """
    Max-and-Attention Pillar Encoding (MAPE)
    Based on the FastPillars (2023) architecture.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard Point-wise Feature Extraction
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        
        # Attention Generation Branch
        self.attention_fc = nn.Linear(out_channels, out_channels, bias=False)
        self.attention_norm = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        # inputs shape: [M, max_points_per_pillar, in_channels]
        # M = total number of non-empty pillars
        
        # 1. Extract Point Features
        x = self.linear(inputs)
        
        # Reshape for BatchNorm1d
        M, num_points, C = x.shape
        x = x.view(M * num_points, C).unsqueeze(-1)
        x = self.norm(x)
        x = x.view(M, num_points, C)
        x = F.relu(x)
        
        # 2. Extract Global Geometric Context (Standard Max Pooling)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # Shape: [M, 1, C]
        
        # 3. Generate Attention Weights
        # Use the global feature to determine which channels matter most
        attn_weights = self.attention_fc(x_max)
        
        # Reshape for BatchNorm
        attn_weights = attn_weights.view(M, C).unsqueeze(-1)
        attn_weights = self.attention_norm(attn_weights)
        attn_weights = attn_weights.view(M, 1, C)
        
        # Sigmoid to scale weights between 0 and 1
        attn_weights = torch.sigmoid(attn_weights)    # Shape: [M, 1, C]
        
        # 4. Attentive Fusion
        # Multiply the original features by the attention weights, then pool
        x_attended = x * attn_weights
        out = torch.max(x_attended, dim=1)[0]         # Shape: [M, C]
        
        return out

class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class FastPillarFeatureNet(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        """
        OpenPCDet version of FastPillars VFE.
        """
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        
        self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)

        if self.use_absolute_xyz:
            num_point_features += 3
        if self.use_cluster_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0

        num_filters = [num_point_features] + list(self.num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            
            if i < len(num_filters) - 2:
                pfn_layers.append(
                    PFNLayer(in_filters, out_filters, use_norm=self.use_norm, last_layer=False)
                )
            else:
                pfn_layers.append(
                    MAPELayer(in_filters, out_filters)
                )
                
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def forward(self, batch_dict, **kwargs):
        voxel_features = batch_dict['voxel_features']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        features = [voxel_features]
        if self.use_cluster_xyz:
            features.append(f_cluster)
        if self.use_absolute_xyz:
            features.append(f_center)
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        features = features.squeeze()
        
        batch_dict['pillar_features'] = features
        return batch_dict