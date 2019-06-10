import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class Resnet50Fc(nn.Module):
    def __init__(self):
        super(Resnet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class Resnet101Fc(nn.Module):
    def __init__(self):
        super(Resnet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Spectral(nn.Module):
    def __init__(self, module, name='weight', iterations=1):
        super(Spectral, self).__init__()
        self.module = module
        self.name = name
        self.iterations = iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        self.sigma = sigma
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class BSP_Res50(nn.Module):
    def __init__(self,num_classes):
        super(BSP_Res50,self).__init__()
        self.model_fc = Resnet50Fc()
        self.bottleneck_layer1 = nn.Linear(2048, 256)
        self.bottleneck_layer1.weight.data.normal_(0, 0.005)
        self.bottleneck_layer1.bias.data.fill_(0.1)
        self.bottleneck_layer1 = Spectral(self.bottleneck_layer1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = nn.Linear(256, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.classifier_layer = Spectral(self.classifier_layer)
        self.predict_layer = nn.Sequential(self.model_fc,self.bottleneck_layer,self.classifier_layer)
        self.ad_layer1 = nn.Linear(256, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5), self.ad_layer2, nn.ReLU(), nn.Dropout(0.5),self.ad_layer3,
                               nn.Sigmoid())
        self.grl = AdversarialLayer(high=1.0)
    def forward(self,x):
        feature = self.model_fc(x)
        out = self.bottleneck_layer(feature)
        outC= self.classifier_layer(out)
        outD = self.ad_net(self.grl(out))
        return(outC,outD,out)

class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                self.high - self.low) + self.low)
        return -coeff * gradOutput





class BSP_Res101(nn.Module):
    def __init__(self,num_classes):
        super(BSP_Res101,self).__init__()
        self.model_fc = Resnet50Fc()
        self.bottleneck_layer1 = nn.Linear(2048, 256)
        self.bottleneck_layer1.weight.data.normal_(0, 0.005)
        self.bottleneck_layer1.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = nn.Linear(256, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)
        self.predict_layer = nn.Sequential(self.model_fc,self.bottleneck_layer,self.classifier_layer)
        self.ad_layer1 = nn.Linear(256, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5), self.ad_layer2, nn.ReLU(), nn.Dropout(0.5),self.ad_layer3,
                               nn.Sigmoid())
        self.grl = AdversarialLayer(high=1.0)
    def forward(self,x):
        feature = self.model_fc(x)
        out = self.bottleneck_layer(feature)
        outC= self.classifier_layer(out)
        outD = self.ad_net(self.grl(out))
        return(outC,outD,out)