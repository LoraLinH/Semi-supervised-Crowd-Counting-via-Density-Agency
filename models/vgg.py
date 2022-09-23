import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
import math
from .transformer import TransformerDecoder, TransformerDecoderLayer


__all__ = ['vgg19', 'vgg19_mask', 'vgg19_mask_up', 'vgg_19_3D', 'vgg19_z', 'vgg19_y', 'vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        h = int(h) // 8
        w = int(w) // 8
        x = self.features(x)
        x = F.upsample_bilinear(x, size=(h, w))
        x = self.reg_layer_0(x)
        return torch.relu(x)

class VGG_Trans(nn.Module):
    def __init__(self, features):
        super(VGG_Trans, self).__init__()
        self.features = features

        d_model = 512
        nhead = 2
        num_layers = 2
        dim_feedforward = 1024
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.decoder = TransformerDecoder(decoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.mask = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.memory_list = nn.Parameter(torch.randn((24, 512)))


    def forward(self, x):

        x = self.features(x)
        mask = torch.sigmoid(self.mask(x))
        mask_de = (mask.detach() > 0.5)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        y = x.clone()
        if mask_de.sum():
            y[mask_de.flatten()] = self.decoder(y[mask_de.flatten()], x)
        z = y.permute(1, 2, 0).view(bs, c, h, w)

        z = self.reg_layer_0(z)
        return torch.relu(z)*mask_de, mask, x.squeeze(1)


class VGG_Mask(nn.Module):
    def __init__(self, features):
        super(VGG_Mask, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.reg_layer_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )


    def forward(self, x):
        x_0 = self.features(x)
        x = self.reg_layer(x_0)
        x_1 = self.reg_layer_1(x_0)
        return x, torch.relu(x_1)


class VGG_3D(nn.Module):
    def __init__(self, features, scale_dim):
        super(VGG_3D, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, scale_dim, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.reg_layer(x)
        return torch.relu(x)   # output is the probability


class VGG_Mask_8(nn.Module):
    def __init__(self, features):
        super(VGG_Mask_8, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.reg_layer_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )


    def forward(self, x):
        b, c, h, w = x.shape
        h = int(h) // 8
        w = int(w) // 8
        x_0 = self.features(x)
        x_0 = F.upsample_bilinear(x_0, size=(h, w))
        x = self.reg_layer(x_0)
        x_1 = self.reg_layer_1(x_0)
        return x, torch.relu(x_1)


class VGG_Z(nn.Module):
    def __init__(self, features):
        super(VGG_Z, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.reg_layer_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )


    def forward(self, x):
        x_0 = self.features(x)
        x = self.reg_layer(x_0)
        x_1 = self.reg_layer_1(x_0)
        return torch.relu(x), torch.sigmoid(x_1)
        # return torch.relu(x), torch.relu(x_1)

class VGG_Y(nn.Module):
    def __init__(self, features):
        super(VGG_Y, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1)
        )

    def forward(self, x):
        X = self.features(x)
        x = self.reg_layer(X)
        x_0, x_1 = torch.split(x, 1, dim=1)
        return torch.relu(x_0), torch.sigmoid(x_1)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def vgg19_trans():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Trans(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def vgg19_mask():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Mask(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def vgg19_z():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Z(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def vgg19_y():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Y(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def vgg19_mask_up():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_Mask_8(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def vgg_19_3D(fea_dim):
    model = VGG_3D(make_layers(cfg['E']), fea_dim)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
