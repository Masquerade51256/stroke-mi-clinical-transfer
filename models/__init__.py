"""
Model definitions for EEG-BCI experiments.

Only the models used in the paper's experiments are registered here:
EEGNet, ADFCNN, and the ShallowConvNet / DeepConvNet backbones used in the
alternative-backbone robustness analysis.
"""

import torch
import torch.nn as nn
from core.registry import MODELS


def register_all_models():
    """
    Register all available models.

    This function should be called at startup to populate the model registry.
    """

    # EEGNet
    try:
        from .EEGNet import EEGNet
        MODELS.register('EEGNet')(EEGNet)
    except ImportError as e:
        print(f"Warning: Could not register EEGNet: {e}")

    # ADFCNN
    try:
        from .myADFCNN import ADFCNN

        # Create adapter to handle multi-band input
        class ADFCNNAdapter(nn.Module):
            def __init__(self, num_channels, num_classes, num_bands, input_length, **kwargs):
                super().__init__()
                self.num_bands = num_bands
                self.num_channels = num_channels

                # ADFCNN expects input shape [B, 1, C, T]
                # But data comes as [B, num_bands, C, T]
                # We reshape to [B, 1, num_bands*C, T]
                self.model = ADFCNN(
                    num_classes=num_classes,
                    num_channels=num_channels * num_bands,  # Merge bands and channels
                    num_bands=1,  # After reshape, we have 1 "band"
                    input_length=input_length
                )

            def forward(self, x):
                # x: [B, num_bands, C, T]
                B = x.shape[0]
                # Reshape to [B, 1, num_bands*C, T]
                # Use reshape instead of view to handle non-contiguous tensors
                x = x.reshape(B, 1, self.num_channels * self.num_bands, -1)
                return self.model(x)

        MODELS.register('ADFCNN')(ADFCNNAdapter)
    except ImportError as e:
        print(f"Warning: Could not register ADFCNN: {e}")

    # ShallowConvNet / DeepConvNet (Schirrmeister et al. 2017)
    # Their file-local classifiers use LogSoftmax + hardcoded kernels, so we
    # wrap the backbones with a LazyLinearWithConstraint head (raw logits,
    # compatible with the trainer's CrossEntropyLoss).
    try:
        from .ShallowConvNet import ShallowConvNet as _ShallowBackbone
        from .DeepConvNet import DeepConvNet as _DeepBackbone
        from .layers import LazyLinearWithConstraint

        class _ConvNetAdapter(nn.Module):
            """Shared adapter: reshape [B, bands, C, T] -> [B, 1, bands*C, T],
            run backbone, classify flattened features with a linear head."""
            backbone_cls = None  # set by subclass

            def __init__(self, num_channels, num_classes, num_bands, input_length,
                         sampling_rate=250, **kwargs):
                super().__init__()
                self.num_bands = num_bands
                self.num_channels = num_channels
                merged_channels = num_channels * num_bands
                if self.backbone_cls is _ShallowBackbone:
                    self.backbone = self.backbone_cls(
                        num_channels=merged_channels, sampling_rate=sampling_rate)
                else:
                    self.backbone = self.backbone_cls(num_channels=merged_channels)

                dummy = torch.randn(1, 1, merged_channels, input_length)
                with torch.no_grad():
                    _, flat = self.backbone(dummy)
                self.classifier = LazyLinearWithConstraint(num_classes, max_norm=0.5)
                # Materialize the lazy layer so parameter counting/optimizer work
                with torch.no_grad():
                    self.classifier(flat)

            def forward(self, x):
                B = x.shape[0]
                x = x.reshape(B, 1, self.num_channels * self.num_bands, -1)
                _, flat = self.backbone(x)
                return self.classifier(flat)

        class ShallowConvNetAdapter(_ConvNetAdapter):
            backbone_cls = _ShallowBackbone

        class DeepConvNetAdapter(_ConvNetAdapter):
            backbone_cls = _DeepBackbone

        MODELS.register('ShallowConvNet')(ShallowConvNetAdapter)
        MODELS.register('DeepConvNet')(DeepConvNetAdapter)
    except ImportError as e:
        print(f"Warning: Could not register ShallowConvNet/DeepConvNet: {e}")


# Auto-register on import
register_all_models()

# Export registry for convenience
__all__ = ['MODELS', 'register_all_models']
