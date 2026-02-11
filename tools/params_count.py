import torch
from monai.networks.nets import BasicUNet, SegResNet, SwinUNETR
import inspect

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def build_basic_unet():
    return BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features=(32, 64, 128, 256, 512, 32),
        dropout=0.1,
    )

def build_segresnet():
    return SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
    )

def build_swinunetr(roi=(96, 96, 96), feature_size=24, use_checkpoint=True):
    sig = inspect.signature(SwinUNETR.__init__)
    params = sig.parameters
    kwargs = dict(
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
    )
    if "use_checkpoint" in params:
        kwargs["use_checkpoint"] = use_checkpoint
    if "img_size" in params:
        kwargs["img_size"] = roi
    return SwinUNETR(**kwargs)

def main():
    models = {
        "BasicUNet3D (32..512)": build_basic_unet(),
        "SegResNet3D (init=32)": build_segresnet(),
        "SwinUNETR3D (fs=24)": build_swinunetr(),
    }

    print("torch:", torch.__version__)
    for name, m in models.items():
        total, trainable = count_params(m)
        print(f"{name:28s} | total: {total/1e6:8.3f} M | trainable: {trainable/1e6:8.3f} M")

if __name__ == "__main__":
    main()
