from .resnet18 import ResNet18
from .mobilenetV2 import MobileNetV2


def get_model(model_name, dataset_name):
    if model_name == 'resnet18':
        return ResNet18()
    elif model_name == 'mobilenetv2':
        return MobileNetV2()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


__all__ = ['get_model', 'ResNet18', 'MobileNetV2']