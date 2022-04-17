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
