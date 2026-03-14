from .unet_3d import UNet3D


def net_factory_3d(net_type="unet_3d", in_chns=1, class_num=2):
    if net_type.lower() != "unet_3d":
        raise ValueError("This simplified project supports only 'unet_3d'.")
    return UNet3D(n_classes=class_num, in_channels=in_chns)
