import torch.nn as nn
import math
from utils import loadPretrain

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        # x = self.fc(x)

        return x


class ResnetReg_norm(nn.Module):

    def __init__(self,modelname='models_facing/resnet18.pth'):
        super(ResnetReg_norm,self).__init__()
        self.encoder = ResNet(BasicBlock, [2, 2, 2, 2])
        loadPretrain(self.encoder, modelname)
        self.reg = nn.Linear(512, 2)
        self.reg.weight.data.normal_(0, 0.01)
        self.reg.bias.data.zero_()

    def forward(self,x):
        x_encode = self.encoder(x)
        x = self.reg(x_encode.view(x_encode.size()[0], -1))
        y = x.abs() # normalize so |x| + |y| = 1
        y = y.sum(dim=1) 
        # import ipdb; ipdb.set_trace()
        x = x/y.unsqueeze(1)
        return x, x_encode

if __name__ == '__main__':
    from facingLabelData import FacingLabelDataset
    from torch.utils.data import DataLoader
    from os.path import join
    from torch.autograd import Variable
    import torchvision.transforms as transforms
    import numpy as np
    np.set_printoptions(precision=2,suppress=False,threshold=100000)
    
    stateEncoder = ResnetReg_norm()
    print stateEncoder

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    datasetdir = '/home/wenshan/datasets'
    imgdataset = FacingLabelDataset(annodir = join(datasetdir,'facing/facing_anno'), 
                                 imgdir=join(datasetdir,'facing/facing_img_coco'), 
                                 imgsize = 128,
                                 mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    dataloader = DataLoader(imgdataset, batch_size=8, shuffle=True, num_workers=1)
    for sample in dataloader:
        inputVar = Variable(sample['img'])
        # print inputVar[0,0,:,:].data.numpy()
        print inputVar.size()
        x, encode = stateEncoder(inputVar)
        print encode.size(), x.size()

        # print encode
        print x