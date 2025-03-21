import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Convolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new


#  First Residual Block
class Convolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output, inputs
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new, inputs


class Convolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


class Convolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output, inputs
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new, inputs


class Convolution5(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution5, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


#  Second Residual Block
class Convolution6(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution6, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, inputs, sel, pr, shortcut_sel, shortcut_pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(inputs, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                shortcut_sel_avg = torch.sum(F.adaptive_avg_pool2d(self.shortcut_weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                shortcut_pr_avg = F.adaptive_avg_pool3d(self.shortcut_weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg, shortcut_sel_avg, shortcut_pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=2, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                shortcut_pr = shortcut_pr.view(self.out_features, 1, 1, 1)
                shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr)
                shortcut_output = nn.functional.conv2d(inputs, shortcut_weights_new, stride=2, padding=1, bias=None)
                shortcut_sel = shortcut_sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                shortcut_output_new = shortcut_output.mul(shortcut_sel)
                shortcut_output_new = self.shortcut_bn(shortcut_output_new)
                return output_new, shortcut_output_new


class Convolution7(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution7, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


class Convolution8(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output, inputs
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new, inputs


class Convolution9(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution9, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


#  Third Residual Block
class Convolution10(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution10, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, inputs, sel, pr, shortcut_sel, shortcut_pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(inputs, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                shortcut_sel_avg = torch.sum(F.adaptive_avg_pool2d(self.shortcut_weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                shortcut_pr_avg = F.adaptive_avg_pool3d(self.shortcut_weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg, shortcut_sel_avg, shortcut_pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=2, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                shortcut_pr = shortcut_pr.view(self.out_features, 1, 1, 1)
                shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr)
                shortcut_output = nn.functional.conv2d(inputs, shortcut_weights_new, stride=2, padding=1, bias=None)
                shortcut_sel = shortcut_sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                shortcut_output_new = shortcut_output.mul(shortcut_sel)
                shortcut_output_new = self.shortcut_bn(shortcut_output_new)
                return output_new, shortcut_output_new


class Convolution11(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution11, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


class Convolution12(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution12, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output, inputs
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new, inputs


class Convolution13(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution13, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


#  Forth Residual Block
class Convolution14(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution14, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, inputs, sel, pr, shortcut_sel, shortcut_pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(inputs, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                shortcut_sel_avg = torch.sum(F.adaptive_avg_pool2d(self.shortcut_weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                shortcut_pr_avg = F.adaptive_avg_pool3d(self.shortcut_weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg, shortcut_sel_avg, shortcut_pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=2, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                shortcut_pr = shortcut_pr.view(self.out_features, 1, 1, 1)
                shortcut_weights_new = self.shortcut_weights.mul(shortcut_pr)
                shortcut_output = nn.functional.conv2d(inputs, shortcut_weights_new, stride=2, padding=1, bias=None)
                shortcut_sel = shortcut_sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                shortcut_output_new = shortcut_output.mul(shortcut_sel)
                shortcut_output_new = self.shortcut_bn(shortcut_output_new)
                return output_new, shortcut_output_new


class Convolution15(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution15, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


class Convolution16(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None)))
            return output, inputs
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = F.relu(self.bn(output_new))
                return output_new, inputs


class Convolution17(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution17, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(self.bias, 0)

    def forward(self, inputs, sel, pr, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(inputs, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                sel_avg = torch.sum(F.adaptive_avg_pool2d(self.weights.view(self.out_features, self.in_features, 3, 3), (1, 1)), dim=1).view(1, self.out_features)
                pr_avg = F.adaptive_avg_pool3d(self.weights.view(self.out_features, 1, self.in_features, 3, 3), (1, 1, 1)).view(1, self.out_features)
                return sel_avg, pr_avg
            elif i == 1:
                pr = pr.view(self.out_features, 1, 1, 1)
                weights_new = self.weights.mul(pr)
                output = nn.functional.conv2d(inputs, weights_new, stride=1, padding=1, bias=None)
                sel = sel.view(1, self.out_features, 1, 1).expand(inputs.size(0), self.out_features, 1, 1)
                output_new = output.mul(sel)
                output_new = self.bn(output_new)
                return output_new


class Fully_Connection(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fully_Connection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weights, 0, 0.01)
        # init.constant_(self.bias, 0)

    def forward(self, inputs, pr, i, pattern):
        if pattern == 0:
            output = inputs.mm(self.weights)
            return output
        else:
            if i == 0:
                pr_avg = F.adaptive_avg_pool2d(self.weights.view(self.in_features, 1, self.out_features), (1, 1)).view(1, self.in_features)
                return pr_avg
            elif i == 1:
                pr = pr.view(self.in_features, 1)
                weights_new = self.weights.mul(pr)
                output = inputs.mm(weights_new)
                return output