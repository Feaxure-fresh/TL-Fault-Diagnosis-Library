import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ADDneck, resnet18


class MFSAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet18(False, in_channel=in_channel)
        self.sonnet1 = ADDneck(512, 256)
        self.sonnet2 = ADDneck(512, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, data_src, data_tgt = 0, label_src = 0, mark = 1):
        mmd_loss = 0.0
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
                mmd_loss += utils.mmd(data_src, data_tgt_son1)

                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)
                pred_src = self.cls_fc_son1(data_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 2:
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)

                data_tgt_son2 = self.sonnet2(data_tgt)
                data_tgt_son2 = self.avgpool(data_tgt_son2)
                data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)

                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss += utils.mmd(data_src, data_tgt_son2)

                data_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

                data_tgt_son1 = self.sonnet1(data_tgt)
                data_tgt_son1 = self.avgpool(data_tgt_son1)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
                l1_loss = torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) - torch.nn.functional.softmax(data_tgt_son2, dim=1))
                l1_loss = torch.mean(l1_loss)

                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

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

if __name__ == '__main__':
    x = torch.randn((32, 1, 1024))
    tar = torch.randn((32, 1, 1024))
    label = torch.randint(3, (32,))
    model = MFSAN()
    model.train()
    cls_loss, mmd_loss, l1_loss = model(x, tar, label)
    print(cls_loss, mmd_loss, l1_loss)