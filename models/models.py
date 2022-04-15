import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


# -----------------------input size>=32---------------------------------
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, out_channel=10, zero_init_residual=False):
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
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

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

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

class DA_Resnet1d(nn.Module):
    
    def __init__(self, in_channel=1, out_channel=10):
        super(DA_Resnet1d, self).__init__()
        self.sharedNet = resnet18(in_channel=in_channel, out_channel=out_channel)
        self.sonnet1 = ADDneck(512, 256)
        self.cls_fc_son1 = nn.Linear(256, out_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, data_src, data_tgt = 0, label_src = 0):
        mmd_loss = 0
        if self.training == True:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd(data_src, data_tgt_son1)
                
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                
                pred = self.cls_fc_son1(data_tgt_son1)
                return cls_loss, mmd_loss, pred
        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred = self.cls_fc_son1(fea_son1)

            return pred
        
class MSFAN(nn.Module):
    
    def __init__(self, in_channel=1, out_channel=10):
        super(MSFAN, self).__init__()
        self.sharedNet = resnet18(in_channel=in_channel, out_channel=out_channel)
        self.sonnet1 = ADDneck(512, 256)
        self.sonnet2 = ADDneck(512, 256)
        self.cls_fc_son1 = nn.Linear(256, out_channel)
        self.cls_fc_son2 = nn.Linear(256, out_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        mmd_loss = 0
        if self.training == True:
            if mark == 1:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                print('data_src:', data_src.shape)
                print('data_tgt_son1:', data_tgt_son1.shape)
                mmd_loss += mmd(data_src, data_tgt_son1)

                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
                
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)
                
                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss, data_tgt_son1

            if mark == 2:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += mmd(data_src, data_tgt_son2)

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
                
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)

                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss, data_tgt_son2
        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            return pred1, pred2


class FeatureExtractor(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_layers,
                 hidden_size,
                 dropout):
        super(FeatureExtractor, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, dropout=dropout, batch_first=True)
        
    def forward(self, input):
        if len(input) == 2:
            source_inputs, target_inputs = input
            
            source_inputs = source_inputs.permute(0, 2, 1)
            output, _ = self.rnn(source_inputs, None)
            
            target_inputs = target_inputs.permute(0, 2, 1)
            target_output, _ = self.rnn(target_inputs, None)
            return output, target_output
        else:
            input = input.permute(0, 2, 1)
            output, _ = self.rnn(input, None)
            return output


class MLP(nn.Module):

    def __init__(self,
                 num_layers,
                 input_size,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False,
                 is_tagger=False):
        super(MLP, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.hidden_size = hidden_size
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))

            if i == 0:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(input_size, hidden_size))
                self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
                hsize = hidden_size
            elif i+1 < num_layers:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
                self.net.add_module('p-relu-{}'.format(i), nn.ReLU())
            else:
                if is_tagger:
                    self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
                    self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))
                else:
                    self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, output_size))
                    hsize = output_size
                    self.net.add_module('p-tanh-{}'.format(i), nn.Tanh())
                    
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.LayerNorm(hsize))
                self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        return self.net(input)
    
    
class MixtureOfExperts(nn.Module):
    
   def __init__(self,
                num_layers,
                input_size,
                hidden_size,
                output_size,
                num_experts,
                dropout,
                signal_length=1024,
                batch_norm=False):
       super(MixtureOfExperts, self).__init__()
       self.signal_length = signal_length
       self.gate = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                                           kernel_size=32, stride=8),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size=int((signal_length-32)/8+1)),
                                 nn.Flatten(),
                                 nn.Linear(in_features=hidden_size, out_features=num_experts),
                                 nn.Softmax(dim=-1))
       self.experts = nn.ModuleList([MLP(num_layers=num_layers, input_size=input_size,
                                         hidden_size=hidden_size, output_size=output_size,
                                         dropout=dropout, batch_norm=batch_norm)
                                     for _ in range(num_experts)])
       
   def forward(self, input):
       if len(input) == 2:
           source_inputs, target_inputs = input
           
           gate_input = source_inputs.detach()
           gate_input = gate_input.permute(0, 2, 1)
           gate_outs = self.gate(gate_input)
    
           expert_outs = torch.stack([exp(source_inputs) for exp in self.experts], dim=-2)
           target_outs = torch.stack([exp(target_inputs) for exp in self.experts], dim=-2)
           
           gate_mult = gate_outs.unsqueeze(1).expand(gate_outs.shape[0], self.signal_length,
                                                     gate_outs.shape[1])
           
           output = torch.sum(gate_mult.unsqueeze(-1) * expert_outs, dim=-2)
           target_out = torch.sum(gate_mult.unsqueeze(-1) * target_outs, dim=-2)
           return output, target_out, gate_outs
       else:
           gate_input = input.detach()
           gate_input = gate_input.permute(0, 2, 1)
           gate_outs = self.gate(gate_input)
           
           expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
           
           gate_mult = gate_outs.unsqueeze(1).expand(gate_outs.shape[0], self.signal_length,
                                                     gate_outs.shape[1])
           output = torch.sum(gate_mult.unsqueeze(-1) * expert_outs, dim=-2)
           return output, gate_outs


class MoE_out(nn.Module):
    
   def __init__(self,
                num_layers,
                input_size,
                hidden_size,
                output_size,
                num_experts,
                dropout,
                signal_length=1024,
                batch_norm=False):
       super(MoE_out, self).__init__()
       self.gate = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                                           kernel_size=32, stride=8),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size=int((signal_length-32)/8+1)),
                                 nn.Flatten(),
                                 nn.Linear(in_features=hidden_size, out_features=num_experts),
                                 nn.Softmax(dim=-1))
       self.experts = nn.ModuleList([
               nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                                                   kernel_size=32, stride=8),
                             nn.ReLU(),
                             nn.MaxPool1d(kernel_size=int((signal_length-32)/8+1)),
                             nn.Flatten(),
                             nn.Linear(in_features=hidden_size, out_features=int(2*hidden_size)),
                             nn.ReLU(),
                             nn.Linear(in_features=int(2*hidden_size), out_features=int(4*hidden_size)),
                             nn.ReLU())
                             for _ in range(num_experts)
                             ])
       self.fc = nn.Sequential(nn.Linear(in_features=int(4*hidden_size), out_features=output_size),
                               nn.LogSoftmax(dim=-1)
                              )
       
   def forward(self, input):
       if len(input) == 2:
           source_feat, target_feat = input
           
           gate_input = source_feat.detach()
           gate_input = gate_input.permute(0, 2, 1)
           gate_outs = self.gate(gate_input)
           
           source_feat = source_feat.permute(0, 2, 1)
           expert_outs = torch.stack([exp(source_feat) for exp in self.experts], dim=-2)
           output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
           
           target_feat = target_feat.permute(0, 2, 1)
           target_outs = torch.stack([exp(target_feat) for exp in self.experts], dim=-2)
           target_output = torch.sum(gate_outs.unsqueeze(-1) * target_outs, dim=-2)
    
           mmd_loss = mmd(output, target_output)
           
           output = self.fc(output)
           return output, gate_outs, mmd_loss
       else:
           gate_input = input.detach()
           gate_input = gate_input.permute(0, 2, 1)
           gate_outs = self.gate(gate_input)
           
           input = input.permute(0, 2, 1)
           expert_outs = torch.stack([exp(input) for exp in self.experts], dim=-2)
           output = torch.sum(gate_outs.unsqueeze(-1) * expert_outs, dim=-2)
                   
           output = self.fc(output)
           return output, gate_outs


class SpAttnMixtureOfExperts(nn.Module):
    
    def __init__(self,
                 num_layers,
                 shared_input_size,
                 private_input_size,
                 num_experts,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SpAttnMixtureOfExperts, self).__init__()
        input_size = shared_input_size + private_input_size
        self.moe = MoE_out(num_layers=num_layers, input_size=input_size, 
                            hidden_size=hidden_size, output_size=output_size,
                            num_experts=num_experts, dropout=dropout,
                            batch_norm=batch_norm)
        
    def forward(self, input):
        if len(input) == 4:
            fs, fp, t_fs, t_fp = input
            features = torch.cat([fs, fp], dim=-1)
            target_features = torch.cat([t_fs, t_fp], dim=-1)
            output = self.moe((features, target_features))
        else:
            fs, fp = input
            features = torch.cat([fs, fp], dim=-1)
            output = self.moe(features)
        return output


class CNNDiscriminator(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 num_src,
                 dropout,
                 signal_length=1024,
                 logsm=True):
        super(CNNDiscriminator, self).__init__()
        window_sizes = [4, 8, 16, 24, 32]
        self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_channels=input_size, 
                                        out_channels=output_size, 
                                        kernel_size=h,
                                        stride=4),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=int((signal_length-h)/4+1)))
                              for h in window_sizes
                              ])
        self.fc = nn.Linear(in_features=output_size*len(window_sizes),
                            out_features=num_src)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input):
        input = input.permute(0, 2, 1)
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1).squeeze(-1)
        
        out = self.fc(out)
        output = self.softmax(out)
        return output


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
