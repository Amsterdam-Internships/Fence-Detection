import config
import segmentation_models_pytorch as smp


def get_encoder(name):
    model = getattr(smp, name)
    model = model(encoder_name=config.ENCODER_DETAILS, 
                  encoder_weights=config.ENCODER_WEIGHTS, 
                  classes=config.CLASSES, 
                  activation=config.ACTIVATION)
    return model


FPN = get_encoder('FPN')

UNet = get_encoder('Unet')

UNetPP = get_encoder('UnetPlusPlus')

PSPNet = get_encoder('PSPNet')

DeepLabV3 = get_encoder('DeepLabV3')

MANet = get_encoder('MAnet')

PAN = get_encoder('PAN')

Linknet = get_encoder('Linknet')