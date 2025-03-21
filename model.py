import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import ViT
from layers import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, \
    Convolution10, Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Fully_Connection


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


class ResNet18(nn.Module):
    def __init__(self, in_channel=3, c_0=64, c_1=128, c_2=256, c_3=512, f_1=512, num_classes=10, expansion=1):
        super(ResNet18, self).__init__()
        self.c_layer1 = Convolution1(in_channel, c_0)
        self.c_layer2 = Convolution2(c_0, c_0)
        self.c_layer3 = Convolution3(c_0, c_0)
        self.c_layer4 = Convolution4(c_0, c_0)
        self.c_layer5 = Convolution5(c_0, c_0)
        self.c_layer6 = Convolution6(c_0, c_1)
        self.c_layer7 = Convolution7(c_1, c_1)
        self.c_layer8 = Convolution8(c_1, c_1)
        self.c_layer9 = Convolution9(c_1, c_1)
        self.c_layer10 = Convolution10(c_1, c_2)
        self.c_layer11 = Convolution11(c_2, c_2)
        self.c_layer12 = Convolution12(c_2, c_2)
        self.c_layer13 = Convolution13(c_2, c_2)
        self.c_layer14 = Convolution14(c_2, c_3)
        self.c_layer15 = Convolution15(c_3, c_3)
        self.c_layer16 = Convolution16(c_3, c_3)
        self.c_layer17 = Convolution17(c_3, c_3)
        self.f_layer18 = Fully_Connection(f_1, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4, 4)

        self.trans1 = nn.Sequential(
            SepConv(
                channel_in=64 * expansion,
                channel_out=128 * expansion
            ),
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans2 = nn.Sequential(
            SepConv(
                channel_in=128 * expansion,
                channel_out=256 * expansion,
            ),
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans3 = nn.Sequential(
            SepConv(
                channel_in=256 * expansion,
                channel_out=512 * expansion,
            ),
            nn.AvgPool2d(4, 4)
        )

        self.trans4 = nn.AvgPool2d(4, 4)

    def forward(self, input, sel_1, pr_1, sel_2, pr_2, sel_3, pr_3, sel_4, pr_4, pattern, i):
        feature_list = []
        feature_list1 = []
        if pattern == 0:
            out = self.c_layer1(input, 0, 0, 0, pattern)
            out, out0 = self.c_layer2(out, 0, 0, 0, pattern)
            out = self.c_layer3(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, 0, 0, 0, pattern)
            out = self.c_layer5(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer6(out, 0, 0, 0, 0, 0, pattern)
            out = self.c_layer7(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, 0, 0, 0, pattern)
            out = self.c_layer9(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer10(out, 0, 0, 0, 0, 0, pattern)
            out = self.c_layer11(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, 0, 0, 0, pattern)
            out = self.c_layer13(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer14(out, 0, 0, 0, 0, 0, pattern)
            out = self.c_layer15(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, 0, 0, 0, pattern)
            out = self.c_layer17(out, 0, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out1_feature = self.trans1(feature_list[0]).view(input.size(0), -1)
            out2_feature = self.trans2(feature_list[1]).view(input.size(0), -1)
            out3_feature = self.trans3(feature_list[2]).view(input.size(0), -1)
            out4_feature = self.trans4(feature_list[3]).view(input.size(0), -1)
            out = self.f_layer18(out4_feature, 0, 0, pattern)
            feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]
            for index in range(len(feat_list)):
                feat_list[index] = F.normalize(feat_list[index], dim=1)
            if self.training:
                return out, feat_list
            else:
                return out
        elif pattern == 1:
            sel1, pr1 = self.c_layer1(input, 0, 0, 0, pattern)
            sel2, pr2 = self.c_layer2(input, 0, 0, 0, pattern)
            sel3, pr3 = self.c_layer3(input, 0, 0, 0, pattern)
            sel4, pr4 = self.c_layer4(input, 0, 0, 0, pattern)
            sel5, pr5 = self.c_layer5(input, 0, 0, 0, pattern)
            sel_1 = torch.cat((sel1, sel2, sel3, sel4, sel5), dim=0)  # [5, 64]
            pr_1 = torch.cat((pr1, pr2, pr3, pr4, pr5), dim=0)  # [5, 64]
            sp_1 = torch.cat((sel_1, pr_1), dim=0)  # [10, 64]
            sel6, pr6, shortcut_sel6, shortcut_pr6 = self.c_layer6(input, 0, 0, 0, 0, 0, pattern)
            sel7, pr7 = self.c_layer7(input, 0, 0, 0, pattern)
            sel8, pr8 = self.c_layer8(input, 0, 0, 0, pattern)
            sel9, pr9 = self.c_layer9(input, 0, 0, 0, pattern)
            sel_2 = torch.cat((sel6, shortcut_sel6, sel7, sel8, sel9), dim=0)  # [5, 128]
            pr_2 = torch.cat((pr6, shortcut_pr6, pr7, pr8, pr9), dim=0)  # [5, 128]
            sp_2 = torch.cat((sel_2, pr_2), dim=0)  # [10, 128]
            sel10, pr10, shortcut_sel10, shortcut_pr10 = self.c_layer10(input, 0, 0, 0, 0, 0, pattern)
            sel11, pr11 = self.c_layer11(input, 0, 0, 0, pattern)
            sel12, pr12 = self.c_layer12(input, 0, 0, 0, pattern)
            sel13, pr13 = self.c_layer13(input, 0, 0, 0, pattern)
            sel_3 = torch.cat((sel10, shortcut_sel10, sel11, sel12, sel13), dim=0)  # [5, 256]
            pr_3 = torch.cat((pr10, shortcut_pr10, pr11, pr12, pr13), dim=0)  # [5, 256]
            sp_3 = torch.cat((sel_3, pr_3), dim=0)  # [10, 256]
            sel14, pr14, shortcut_sel14, shortcut_pr14 = self.c_layer14(input, 0, 0, 0, 0, 0, pattern)
            sel15, pr15 = self.c_layer15(input, 0, 0, 0, pattern)
            sel16, pr16 = self.c_layer16(input, 0, 0, 0, pattern)
            sel17, pr17 = self.c_layer17(input, 0, 0, 0, pattern)
            pr18 = self.f_layer18(input, 0, 0, pattern)
            sel_4 = torch.cat((sel14, shortcut_sel14, sel15, sel16, sel17), dim=0)  # [5, 512]
            pr_4 = torch.cat((pr14, shortcut_pr14, pr15, pr16, pr17, pr18), dim=0)  # [6, 512]
            sp_4 = torch.cat((sel_4, pr_4), dim=0)  # [11, 512]
            return sp_1, sp_2, sp_3, sp_4
        elif pattern == 2:
            if i == 1 or i == 2 or i == 3 or i == 4:
                sel1 = sel_1[0:1, :]
                sel2 = sel_1[1:2, :]
                sel3 = sel_1[2:3, :]
                sel4 = sel_1[3:4, :]
                sel5 = sel_1[4:5, :]
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
############################################
                sel6 = sel_2[0:1, :]
                shortcut_sel6 = sel_2[1:2, :]
                sel7 = sel_2[2:3, :]
                sel8 = sel_2[3:4, :]
                sel9 = sel_2[4:5, :]
                pr6 = pr_2[0:1, :]
                shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
############################################
                sel10 = sel_3[0:1, :]
                shortcut_sel10 = sel_3[1:2, :]
                sel11 = sel_3[2:3, :]
                sel12 = sel_3[3:4, :]
                sel13 = sel_3[4:5, :]
                pr10 = pr_3[0:1, :]
                shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
############################################
                sel14 = sel_4[0:1, :]
                shortcut_sel14 = sel_4[1:2, :]
                sel15 = sel_4[2:3, :]
                sel16 = sel_4[3:4, :]
                sel17 = sel_4[4:5, :]
                pr14 = pr_4[0:1, :]
                shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                return out
            elif i == 5:
                an1_lstm1, an1_lstm2 = pr_1
                data1_an1_lstm1, data2_an1_lstm1 = an1_lstm1
                data1_an1_lstm2, data2_an1_lstm2 = an1_lstm2
                sel1 = sel_1[0:1, :]
                sel2 = sel_1[1:2, :]
                sel3 = sel_1[2:3, :]
                sel4 = sel_1[3:4, :]
                sel5 = sel_1[4:5, :]
                pr1 = data1_an1_lstm1[0:1, :]
                pr2 = data1_an1_lstm1[1:2, :]
                pr3 = data1_an1_lstm1[2:3, :]
                pr4 = data1_an1_lstm1[3:4, :]
                pr5 = data1_an1_lstm1[4:5, :]
############################################
                sel6 = sel_2[0:1, :]
                shortcut_sel6 = sel_2[1:2, :]
                sel7 = sel_2[2:3, :]
                sel8 = sel_2[3:4, :]
                sel9 = sel_2[4:5, :]
                pr6 = pr_2[0:1, :]
                shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
############################################
                sel10 = sel_3[0:1, :]
                shortcut_sel10 = sel_3[1:2, :]
                sel11 = sel_3[2:3, :]
                sel12 = sel_3[3:4, :]
                sel13 = sel_3[4:5, :]
                pr10 = pr_3[0:1, :]
                shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
############################################
                sel14 = sel_4[0:1, :]
                shortcut_sel14 = sel_4[1:2, :]
                sel15 = sel_4[2:3, :]
                sel16 = sel_4[3:4, :]
                sel17 = sel_4[4:5, :]
                pr14 = pr_4[0:1, :]
                shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                pr1 = data2_an1_lstm1[0:1, :]
                pr2 = data2_an1_lstm1[1:2, :]
                pr3 = data2_an1_lstm1[2:3, :]
                pr4 = data2_an1_lstm1[3:4, :]
                pr5 = data2_an1_lstm1[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                pr1 = data1_an1_lstm2[0:1, :]
                pr2 = data1_an1_lstm2[1:2, :]
                pr3 = data1_an1_lstm2[2:3, :]
                pr4 = data1_an1_lstm2[3:4, :]
                pr5 = data1_an1_lstm2[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out1 = self.f_layer18(out, pr18, 1, pattern)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out1, feature_list1
                else:
                    return out1
            elif i == 6:
                an2_lstm1, an2_lstm2 = pr_2
                data1_an2_lstm1, data2_an2_lstm1 = an2_lstm1
                data1_an2_lstm2, data2_an2_lstm2 = an2_lstm2
                sel1 = sel_1[0:1, :]
                sel2 = sel_1[1:2, :]
                sel3 = sel_1[2:3, :]
                sel4 = sel_1[3:4, :]
                sel5 = sel_1[4:5, :]
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
############################################
                sel6 = sel_2[0:1, :]
                shortcut_sel6 = sel_2[1:2, :]
                sel7 = sel_2[2:3, :]
                sel8 = sel_2[3:4, :]
                sel9 = sel_2[4:5, :]
                pr6 = data1_an2_lstm1[0:1, :]
                shortcut_pr6 = data1_an2_lstm1[1:2, :]
                pr7 = data1_an2_lstm1[2:3, :]
                pr8 = data1_an2_lstm1[3:4, :]
                pr9 = data1_an2_lstm1[4:5, :]
############################################
                sel10 = sel_3[0:1, :]
                shortcut_sel10 = sel_3[1:2, :]
                sel11 = sel_3[2:3, :]
                sel12 = sel_3[3:4, :]
                sel13 = sel_3[4:5, :]
                pr10 = pr_3[0:1, :]
                shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
############################################
                sel14 = sel_4[0:1, :]
                shortcut_sel14 = sel_4[1:2, :]
                sel15 = sel_4[2:3, :]
                sel16 = sel_4[3:4, :]
                sel17 = sel_4[4:5, :]
                pr14 = pr_4[0:1, :]
                shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                pr6 = data2_an2_lstm1[0:1, :]
                shortcut_pr6 = data2_an2_lstm1[1:2, :]
                pr7 = data2_an2_lstm1[2:3, :]
                pr8 = data2_an2_lstm1[3:4, :]
                pr9 = data2_an2_lstm1[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                pr6 = data1_an2_lstm2[0:1, :]
                shortcut_pr6 = data1_an2_lstm2[1:2, :]
                pr7 = data1_an2_lstm2[2:3, :]
                pr8 = data1_an2_lstm2[3:4, :]
                pr9 = data1_an2_lstm2[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out1 = self.f_layer18(out, pr18, 1, pattern)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out1, feature_list1
                else:
                    return out1
            elif i == 7:
                an3_lstm1, an3_lstm2 = pr_3
                data1_an3_lstm1, data2_an3_lstm1 = an3_lstm1
                data1_an3_lstm2, data2_an3_lstm2 = an3_lstm2
                sel1 = sel_1[0:1, :]
                sel2 = sel_1[1:2, :]
                sel3 = sel_1[2:3, :]
                sel4 = sel_1[3:4, :]
                sel5 = sel_1[4:5, :]
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
############################################
                sel6 = sel_2[0:1, :]
                shortcut_sel6 = sel_2[1:2, :]
                sel7 = sel_2[2:3, :]
                sel8 = sel_2[3:4, :]
                sel9 = sel_2[4:5, :]
                pr6 = pr_2[0:1, :]
                shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
############################################
                sel10 = sel_3[0:1, :]
                shortcut_sel10 = sel_3[1:2, :]
                sel11 = sel_3[2:3, :]
                sel12 = sel_3[3:4, :]
                sel13 = sel_3[4:5, :]
                pr10 = data1_an3_lstm1[0:1, :]
                shortcut_pr10 = data1_an3_lstm1[1:2, :]
                pr11 = data1_an3_lstm1[2:3, :]
                pr12 = data1_an3_lstm1[3:4, :]
                pr13 = data1_an3_lstm1[4:5, :]
############################################
                sel14 = sel_4[0:1, :]
                shortcut_sel14 = sel_4[1:2, :]
                sel15 = sel_4[2:3, :]
                sel16 = sel_4[3:4, :]
                sel17 = sel_4[4:5, :]
                pr14 = pr_4[0:1, :]
                shortcut_pr14 = pr_4[1:2, :]
                pr15 = pr_4[2:3, :]
                pr16 = pr_4[3:4, :]
                pr17 = pr_4[4:5, :]
                pr18 = pr_4[5:6, :]
                # 1
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                pr10 = data2_an3_lstm1[0:1, :]
                shortcut_pr10 = data2_an3_lstm1[1:2, :]
                pr11 = data2_an3_lstm1[2:3, :]
                pr12 = data2_an3_lstm1[3:4, :]
                pr13 = data2_an3_lstm1[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                pr10 = data1_an3_lstm2[0:1, :]
                shortcut_pr10 = data1_an3_lstm2[1:2, :]
                pr11 = data1_an3_lstm2[2:3, :]
                pr12 = data1_an3_lstm2[3:4, :]
                pr13 = data1_an3_lstm2[4:5, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out1 = self.f_layer18(out, pr18, 1, pattern)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out1, feature_list1
                else:
                    return out1
            elif i == 8:
                an4_lstm1, an4_lstm2 = pr_4
                data1_an4_lstm1, data2_an4_lstm1 = an4_lstm1
                data1_an4_lstm2, data2_an4_lstm2 = an4_lstm2
                sel1 = sel_1[0:1, :]
                sel2 = sel_1[1:2, :]
                sel3 = sel_1[2:3, :]
                sel4 = sel_1[3:4, :]
                sel5 = sel_1[4:5, :]
                pr1 = pr_1[0:1, :]
                pr2 = pr_1[1:2, :]
                pr3 = pr_1[2:3, :]
                pr4 = pr_1[3:4, :]
                pr5 = pr_1[4:5, :]
############################################
                sel6 = sel_2[0:1, :]
                shortcut_sel6 = sel_2[1:2, :]
                sel7 = sel_2[2:3, :]
                sel8 = sel_2[3:4, :]
                sel9 = sel_2[4:5, :]
                pr6 = pr_2[0:1, :]
                shortcut_pr6 = pr_2[1:2, :]
                pr7 = pr_2[2:3, :]
                pr8 = pr_2[3:4, :]
                pr9 = pr_2[4:5, :]
############################################
                sel10 = sel_3[0:1, :]
                shortcut_sel10 = sel_3[1:2, :]
                sel11 = sel_3[2:3, :]
                sel12 = sel_3[3:4, :]
                sel13 = sel_3[4:5, :]
                pr10 = pr_3[0:1, :]
                shortcut_pr10 = pr_3[1:2, :]
                pr11 = pr_3[2:3, :]
                pr12 = pr_3[3:4, :]
                pr13 = pr_3[4:5, :]
############################################
                sel14 = sel_4[0:1, :]
                shortcut_sel14 = sel_4[1:2, :]
                sel15 = sel_4[2:3, :]
                sel16 = sel_4[3:4, :]
                sel17 = sel_4[4:5, :]
                pr14 = data1_an4_lstm1[0:1, :]
                shortcut_pr14 = data1_an4_lstm1[1:2, :]
                pr15 = data1_an4_lstm1[2:3, :]
                pr16 = data1_an4_lstm1[3:4, :]
                pr17 = data1_an4_lstm1[4:5, :]
                pr18 = data1_an4_lstm1[5:6, :]
                # 1
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 2
                pr14 = data2_an4_lstm1[0:1, :]
                shortcut_pr14 = data2_an4_lstm1[1:2, :]
                pr15 = data2_an4_lstm1[2:3, :]
                pr16 = data2_an4_lstm1[3:4, :]
                pr17 = data2_an4_lstm1[4:5, :]
                pr18 = data2_an4_lstm1[5:6, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out = self.f_layer18(out, pr18, 1, pattern)
                feature_list.append(out)
                # 3
                pr14 = data1_an4_lstm2[0:1, :]
                shortcut_pr14 = data1_an4_lstm2[1:2, :]
                pr15 = data1_an4_lstm2[2:3, :]
                pr16 = data1_an4_lstm2[3:4, :]
                pr17 = data1_an4_lstm2[4:5, :]
                pr18 = data1_an4_lstm2[5:6, :]
                out = self.c_layer1(input, sel1, pr1, 1, pattern)
                out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
                out = self.c_layer3(out, sel3, pr3, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
                out = self.c_layer5(out, sel5, pr5, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
                out = self.c_layer7(out, sel7, pr7, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
                out = self.c_layer9(out, sel9, pr9, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
                out = self.c_layer11(out, sel11, pr11, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
                out = self.c_layer13(out, sel13, pr13, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
                out = self.c_layer15(out, sel15, pr15, 1, pattern)
                out += out0
                out = self.relu(out)
                out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
                out = self.c_layer17(out, sel17, pr17, 1, pattern)
                out += out0
                out = self.relu(out)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                out1 = self.f_layer18(out, pr18, 1, pattern)
                feature_list1.append(torch.cat((feature_list[0], feature_list[1]), dim=0))
                for index in range(len(feature_list1)):
                    feature_list1[index] = F.normalize(feature_list1[index], dim=1)
                if self.training:
                    return out1, feature_list1
                else:
                    return out1
        elif pattern == 3:
            sel1 = sel_1[0:1, :]
            sel2 = sel_1[1:2, :]
            sel3 = sel_1[2:3, :]
            sel4 = sel_1[3:4, :]
            sel5 = sel_1[4:5, :]
            pr1 = pr_1[0:1, :]
            pr2 = pr_1[1:2, :]
            pr3 = pr_1[2:3, :]
            pr4 = pr_1[3:4, :]
            pr5 = pr_1[4:5, :]
############################################
            sel6 = sel_2[0:1, :]
            shortcut_sel6 = sel_2[1:2, :]
            sel7 = sel_2[2:3, :]
            sel8 = sel_2[3:4, :]
            sel9 = sel_2[4:5, :]
            pr6 = pr_2[0:1, :]
            shortcut_pr6 = pr_2[1:2, :]
            pr7 = pr_2[2:3, :]
            pr8 = pr_2[3:4, :]
            pr9 = pr_2[4:5, :]
############################################
            sel10 = sel_3[0:1, :]
            shortcut_sel10 = sel_3[1:2, :]
            sel11 = sel_3[2:3, :]
            sel12 = sel_3[3:4, :]
            sel13 = sel_3[4:5, :]
            pr10 = pr_3[0:1, :]
            shortcut_pr10 = pr_3[1:2, :]
            pr11 = pr_3[2:3, :]
            pr12 = pr_3[3:4, :]
            pr13 = pr_3[4:5, :]
############################################
            sel14 = sel_4[0:1, :]
            shortcut_sel14 = sel_4[1:2, :]
            sel15 = sel_4[2:3, :]
            sel16 = sel_4[3:4, :]
            sel17 = sel_4[4:5, :]
            pr14 = pr_4[0:1, :]
            shortcut_pr14 = pr_4[1:2, :]
            pr15 = pr_4[2:3, :]
            pr16 = pr_4[3:4, :]
            pr17 = pr_4[4:5, :]
            pr18 = pr_4[5:6, :]
            out = self.c_layer1(input, sel1, pr1, 1, pattern)
            out, out0 = self.c_layer2(out, sel2, pr2, 1, pattern)
            out = self.c_layer3(out, sel3, pr3, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, sel4, pr4, 1, pattern)
            out = self.c_layer5(out, sel5, pr5, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer6(out, sel6, pr6, shortcut_sel6, shortcut_pr6, 1, pattern)
            out = self.c_layer7(out, sel7, pr7, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, sel8, pr8, 1, pattern)
            out = self.c_layer9(out, sel9, pr9, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer10(out, sel10, pr10, shortcut_sel10, shortcut_pr10, 1, pattern)
            out = self.c_layer11(out, sel11, pr11, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, sel12, pr12, 1, pattern)
            out = self.c_layer13(out, sel13, pr13, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out, out0 = self.c_layer14(out, sel14, pr14, shortcut_sel14, shortcut_pr14, 1, pattern)
            out = self.c_layer15(out, sel15, pr15, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, sel16, pr16, 1, pattern)
            out = self.c_layer17(out, sel17, pr17, 1, pattern)
            out += out0
            out = self.relu(out)
            feature_list.append(out)
            out1_feature = self.trans1(feature_list[0]).view(input.size(0), -1)
            out2_feature = self.trans2(feature_list[1]).view(input.size(0), -1)
            out3_feature = self.trans3(feature_list[2]).view(input.size(0), -1)
            out4_feature = self.trans4(feature_list[3]).view(input.size(0), -1)
            out = self.f_layer18(out4_feature, pr18, 1, pattern)
            feat_list = [out4_feature, out3_feature, out2_feature, out1_feature]
            for index in range(len(feat_list)):
                feat_list[index] = F.normalize(feat_list[index], dim=1)
            if self.training:
                return out, feat_list
            else:
                return out