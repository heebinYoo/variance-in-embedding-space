import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class Feature(nn.Module):
    def __init__(self, model='resnet50', pool='avg', use_lnorm=False):
        nn.Module.__init__(self)
        self.model = model

        self.base = models.__dict__[model](pretrained=True)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise Exception('pool: %s pool must be avg or max', str(pool))

        self.lnorm = None
        if use_lnorm:
            self.lnorm = nn.LayerNorm(2048, elementwise_affine=False).cuda()

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x1 = self.pool(x)
        x = x1
        x = x.reshape(x.size(0), -1)

        if self.lnorm != None:
            x = self.lnorm(x)

        return x

class Feat_resnet50_max(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='max')

class Feat_resnet50_avg(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='avg')

class Feat_resnet50_max_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='max', use_lnorm=True)

class Feat_resnet50_avg_n(Feature):
    def __init__(self):
        Feature.__init__(self, model='resnet50', pool='avg', use_lnorm=True)



class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, 2*n_classes)

    def forward(self, x, aug_sample):
        if len(aug_sample) != 0 :
            aug_output =  self.nonlinear(aug_sample)
            aug_score =  F.log_softmax(self.fc1(aug_output), dim=-1)
            return aug_score

        emb = self.embedding_net(x)
        output = self.nonlinear(emb)

        scores = F.log_softmax(self.fc1(output), dim=-1)

        return scores,emb

    def get_embedding(self, x):
        return self.embedding_net(x)

    def get_weight(self):
        return self.fc1.weight


