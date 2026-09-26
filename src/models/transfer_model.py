import torch.nn as nn
from torchvision import models

def build_transfer_model(model_name, num_classes, freeze_backbone=True):
    # loads a pretrained model and swaps its final layer to match our classes

    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif model_name == "swin_t":
        model = models.swin_t(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        # convnext's classifier is LayerNorm -> Flatten -> Linear, so only swap the last piece
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Unknown model name: " + model_name)

    return model