import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=None,
                 dropout=0,
                 num_layer=3,
                 use_batchnorm=False,
                 out_inter_feat=-1,
                 output_layer=True,
                 last='tanh',
                 size_reduction=4):
        super(MLP, self).__init__()
        
        assert out_inter_feat < num_layer - 1, "out_inter_feat must be less than num_layer - 1"
        layers, final_layers = [], []
        self.out_inter_feat = out_inter_feat
        
        # Determine the size reduction factor for each layer
        size_reduction_factor = size_reduction
        
        current_size = input_size
        self.intermediate_layers = nn.ModuleList()
        
        for i in range(num_layer-1 if output_layer else num_layer):
            next_size = int(current_size / size_reduction_factor)
            layers.append(nn.Linear(current_size, next_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_size = next_size
            if self.out_inter_feat == i or (not output_layer):
                self.feature_dim = current_size
            
            self.intermediate_layers.append(nn.Sequential(*layers))
            layers = []  # Reset layers for the next iteration
        
        if output_layer:
            # Final layer to output size
            final_layers.append(nn.Linear(current_size, output_size))

            # Define the last layer activation
            if last == 'logsm':
                final_layers.append(nn.LogSoftmax(dim=-1))
            elif last == 'sm':
                final_layers.append(nn.Softmax(dim=-1))
            elif last == 'tanh':
                final_layers.append(nn.Tanh())
            elif last == 'sigmoid':
                final_layers.append(nn.Sigmoid())
            elif last == 'relu':
                final_layers.append(nn.ReLU())
        else:
            final_layers.append(nn.Identity())
        
        self.final_layer = nn.Sequential(*final_layers)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.intermediate_layers):
            x = layer(x)
            if self.out_inter_feat == i:
                inter_feat = x
        output = self.final_layer(x)
        if self.out_inter_feat >= 0:
            return inter_feat, output
        else:
            return output
    

class CNN(nn.Module):
    def __init__(self,
                 in_channel=1, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1,
                 mp_kernel_size=2, 
                 mp_stride=2,
                 num_layer=5):
        super(CNN, self).__init__()
        
        layers = []
        current_in_channels = in_channel
        predefined_output_channels = [4, 16, 32, 64, 128]
        num_predefined_layers = len(predefined_output_channels)

        for i in range(num_layer):
            if i < num_predefined_layers:
                current_out_channels = predefined_output_channels[i]
            else:
                current_out_channels = 128  # Use 128 for any additional layers
            
            layers.append(nn.Conv1d(current_in_channels, current_out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding))
            layers.append(nn.BatchNorm1d(current_out_channels))
            layers.append(nn.ReLU(inplace=True))
            if i < num_layer - 1:  # For all but the last layer
                layers.append(nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))
            current_in_channels = current_out_channels

        layers.append(nn.AdaptiveMaxPool1d(1))  # Final adaptive max pool
        layers.append(nn.Flatten())  # Flatten for the output

        self.net = nn.Sequential(*layers)
        self.out_dim = current_out_channels

    def forward(self, input):
        output = self.net(input)
        return output


class MSCNN(nn.Module):
    # multi-scale convolutional neural network
    def __init__(self, in_channel, kernel_sizes=[4, 8, 16, 24, 32], block=CNN):
        super(MSCNN, self).__init__()
        
        # Create multiple CNN blocks with different kernel sizes
        self.convs = nn.ModuleList([
            block(in_channel=in_channel, kernel_size=k)
            for k in kernel_sizes
        ])
        self.out_dim = sum([b.out_dim for b in self.convs])
        self.fl = nn.Flatten()

    def forward(self, input):
        # Apply each CNN block to the input and collect the outputs
        h = [conv(input) for conv in self.convs]
        # Concatenate the outputs along the channel dimension
        h = torch.cat(h, dim=1)
        # Flatten the concatenated output
        output = self.fl(h)

        return output


def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, in_channel, layers, block=BasicBlock, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # output (512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        self.out_dim = planes
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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

        return x


class ClassifierBase(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 backbone='CNN',
                 dropout=0,
                 num_layer=3,
                 use_batchnorm=False,
                 use_cls_feat=-1):
        super(ClassifierBase, self).__init__()
        if backbone == 'CNN':
            self.G = MSCNN(in_channel=input_size)
        elif backbone == 'ResNet':
            self.G = ResNet(in_channel=1, layers=[2, 2, 2, 2])
        else:
            raise Exception(f"unknown backbone type {backbone}")
        self.C = MLP(self.G.out_dim, num_classes, dropout, num_layer, 
                     use_batchnorm, out_inter_feat=use_cls_feat, last=None)
        self.use_cls_feat = use_cls_feat
        if use_cls_feat >= 0:
            self.feature_dim = self.C.feature_dim
        else:
            self.feature_dim = self.G.out_dim
        
    def forward(self, input):
        h = self.G(input)
        if self.use_cls_feat >= 0:
            h, predictions = self.C(h)
        else:
            predictions = self.C(h)
        if self.training:
            return predictions, h
        else:
            return predictions
