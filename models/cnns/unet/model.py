import config

import segmentation_models_pytorch as smp


UNet = smp.Unet(
    encoder_name=config.ENCODER, 
    encoder_weights=config.ENCODER_WEIGHTS, 
    classes=config.CLASSES, 
    activation=config.ACTIVATION,
)