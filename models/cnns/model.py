import config
import segmentation_models_pytorch as smp


def get_encoder(name):
    model = getattr(smp, name)
    model = model(encoder_name=config.ENCODER, 
                  encoder_weights=config.ENCODER_WEIGHTS, 
                  classes=config.CLASSES, 
                  activation=config.ACTIVATION,)
    return model


FPN = get_encoder('FPN')

UNet = get_encoder('Unet')

UNetPP = get_encoder('UnetPlusPlus')

PSPNet = get_encoder('PSPNet')