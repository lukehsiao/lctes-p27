import torch
from torchvision import models


def get_cnn(net_name, pretrained=True, num_classes=1, drop_rate=0.2):
    net = getattr(models, net_name)
    if "densenet" in net_name:
        model = net(pretrained=pretrained, drop_rate=drop_rate)
        last_layer_input_size = int(model.classifier.weight.size()[1])
        model.classifier = torch.nn.Linear(last_layer_input_size, num_classes)
    elif "resnet" in net_name:
        model = net(pretrained=pretrained)
        last_layer_input_size = int(model.fc.weight.size()[1])
        model.fc = torch.nn.Linear(last_layer_input_size, num_classes)
    elif "vgg" in net_name or "alex" in net_name:
        model = net(pretrained=pretrained)
        last_layer_input_size = 4096
        no_last_layer = torch.nn.Sequential(*list(model.classifier.children())[:-1])
        classifier = torch.nn.Sequential(
            no_last_layer, torch.nn.Linear(last_layer_input_size, num_classes)
        )
        model.classifier = classifier
    elif "squeezenet" in net_name:
        model = net(pretrained=pretrained)
        model.num_classes = num_classes
        model.final_conv = torch.nn.Conv2d(512, num_classes, kernel_size=1)
        model.init.normal_(model.final_conv, mean=0.0, std=0.01)
    elif "inception" in net_name:
        model = net(pretrained=pretrained)
        last_layer_input_size = 2048
        model.fc = torch.nn.Linear(last_layer_input_size, num_classes)
    else:
        raise ValueError("Unknown network!")
    return model
