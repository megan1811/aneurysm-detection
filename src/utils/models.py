import torch
import torch.nn as nn


class CoordMLP(nn.Module):
    """
    Small MLP that embeds normalized 3D patch-center coordinates into a learned
    feature vector
    """

    def __init__(self, in_dim=3, hidden_dim=32, out_dim=32):
        """
        Initialize the coordinate MLP Head.

        Args:
            in_dim (int, optional): Input dimension of the coordinates
                (default is 3 for LPS coordinates).
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 32.
            out_dim (int, optional): Output embedding dimension for the
                coordinates. Defaults to 32.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords) -> torch.Tensor:
        """
        Embed raw 3D coordinates into a learned feature space.

        Args:
            coords (torch.Tensor): Tensor of shape (B, 3) containing
                normalized patch-center coordinates scaled to [-1, 1].

        Returns:
            torch.Tensor: Tensor of shape (B, out_dim) containing the
                coordinate embeddings.
        """
        return self.net(coords)


class PatchClassifier(nn.Module):
    """
    Two-head 3D patch model for aneurysm detection and anatomical localization.

    The model encodes a 3D patch with a 3D CNN backbone (MedicalNet ResNet-10),
    then fuses:
      - pooled image features,
      - a learned embedding of the acquisition modality (categorical index),
      - a learned embedding of the patch center coordinates (continuous, scaled to [-1, 1]).

    A shared fusion MLP produces a compact representation that feeds two task heads:
      1) Presence head: binary aneurysm-present logit (trained on all patches).
      2) Location head: logits over aneurysm anatomical location categories
         (intended to be trained only on positive patches via masking).
    """

    def __init__(
        self,
        modality_num_classes: int,
        num_loc_classes: int,
        coord_emb_dim: int = 32,
        modality_emb_dim: int = 16,
        fusion_hidden_dim: int = 256,
        fusion_out_dim: int = 128,
        dropout_p: float = 0.3,
    ):
        """
        Initialize the PatchClassifier.

        Args:
            modality_num_classes (int): Number of distinct imaging modalities.
            num_loc_classes (int): Number of anatomical location categories.
            coord_emb_dim (int, optional): Output embedding dimension of the coordinate MLP. Defaults to 32.
            modality_emb_dim (int, optional): Embedding dimension for modality. Defaults to 16.
            fusion_hidden_dim (int, optional): Hidden dimension of fusion trunk. Defaults to 256.
            fusion_out_dim (int, optional): Output dimension of shared fused embedding. Defaults to 128.
            dropout_p (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()

        self.backbone = torch.hub.load(
            "Warvito/MedicalNet-models", "medicalnet_resnet10_23datasets"
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        backbone_embedding_dim = 512  # output feature dim of the backbone

        self.modality_embedding = nn.Embedding(modality_num_classes, modality_emb_dim)
        self.coord_mlp = CoordMLP(in_dim=3, hidden_dim=32, out_dim=coord_emb_dim)

        # Shared fusion trunk
        fusion_in_dim = backbone_embedding_dim + modality_emb_dim + coord_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(fusion_hidden_dim, fusion_out_dim),
            nn.ReLU(),
        )

        # Heads
        self.head_presence = nn.Linear(fusion_out_dim, 1)  # (B, 1)
        self.head_location = nn.Linear(fusion_out_dim, num_loc_classes)  # (B, 13)

    def forward(
        self,
        patch_volume: torch.Tensor,
        center_coords: torch.Tensor,
        modality_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Patch Classifier forward pass.

        Args:
            patch_volume (torch.Tensor): Tensor of shape (B, 1, D, H, W).
            center_coords (torch.Tensor): Tensor of shape (B, 3), float32, scaled to [-1, 1].
            modality_idx (torch.Tensor): Tensor of shape (B,), dtype long.

        Returns:
            tuple:
                - pres_logits (torch.Tensor): Tensor of shape (B,) containing presence logits.
                - loc_logits (torch.Tensor): Tensor of shape (B, num_loc_classes) containing location logits.
        """

        x = self.backbone(patch_volume)
        x = self.pool(x).view(x.size(0), -1)  # shape (B, C)

        m = self.modality_embedding(modality_idx)  # (B, 16)
        c = self.coord_mlp(center_coords)  # (B, coor_emb_dim)

        h = torch.cat([x, m, c], dim=1)  # (B, fusion_in_dim)
        h = self.fusion(h)  # (B, fusion_out_dim)

        pres_logits = self.head_presence(h).squeeze(1)  # (B,)
        loc_logits = self.head_location(h)  # (B, 13)

        return pres_logits, loc_logits
