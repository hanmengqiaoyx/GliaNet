import torch
import torch.nn as nn
import torch.nn.functional as F
from ViT import ViT


class MLP_Encoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class MLP_Decoder(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Pre_head(nn.Module):
    def __init__(self, image_height=41, image_width=512, patch_height=1, patch_width=512, depth=2, heads=2, dim=512, dim_head=32, mlp_dim=512, channels=1):
        super(Pre_head, self).__init__()
        self.num_classes = image_height
        self.c_1 = 64
        self.c_2 = 128
        self.c_3 = 256
        self.c_4 = 512
        self.encoder1 = MLP_Encoder(self.c_1, self.c_4)
        self.encoder2 = MLP_Encoder(self.c_2, self.c_4)
        self.encoder3 = MLP_Encoder(self.c_3, self.c_4)
        self.encoder4 = MLP_Encoder(self.c_4, self.c_4)

        self.vit = ViT(image_height, image_width, patch_height, patch_width, depth, heads, dim, dim_head, mlp_dim, channels, self.num_classes)

        self.decoder1 = MLP_Decoder(self.c_4, self.c_1)
        self.decoder2 = MLP_Decoder(self.c_4, self.c_2)
        self.decoder3 = MLP_Decoder(self.c_4, self.c_3)
        self.decoder4 = MLP_Decoder(self.c_4, self.c_4)

    def forward(self, weight_avg_1, weight_avg_2, weight_avg_3, weight_avg_4):
        weight_1 = self.encoder1(weight_avg_1)
        weight_2 = self.encoder2(weight_avg_2)
        weight_3 = self.encoder3(weight_avg_3)
        weight_4 = self.encoder4(weight_avg_4)
        weight_avg = torch.cat((weight_1, weight_2, weight_3, weight_4), dim=0).view(1, 1, -1, self.c_4)  # [1, 1, 41, 512]

        preds = self.vit(weight_avg).view(-1, 512)  # [41, 512]

        weight_1 = self.decoder1(preds[0:10])
        weight_2 = self.decoder2(preds[10:20])
        weight_3 = self.decoder3(preds[20:30])
        weight_4 = self.decoder4(preds[30:41])
        weight1_1 = F.normalize(weight_1[0:5], dim=1)
        weight1_2 = F.normalize(weight_1[5:10], dim=1)
        weight2_1 = F.normalize(weight_2[0:5], dim=1)
        weight2_2 = F.normalize(weight_2[5:10], dim=1)
        weight3_1 = F.normalize(weight_3[0:5], dim=1)
        weight3_2 = F.normalize(weight_3[5:10], dim=1)
        weight4_1 = F.normalize(weight_4[0:5], dim=1)
        weight4_2 = F.normalize(weight_4[5:11], dim=1)
        return weight1_1, weight1_2, weight2_1, weight2_2, weight3_1, weight3_2, weight4_1, weight4_2


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=False, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
             cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
             cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class SepConv_ReLU(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(SepConv_ReLU, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)


class Oli_Network_1(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Oli_Network_1, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.classifier = nn.Sequential(
            SepConv_ReLU(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg):
        weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
        sel, state = self.convlstm(weights)
        sel = torch.squeeze(torch.squeeze(torch.stack(sel), dim=0), dim=0)
        sel = self.classifier(sel).view(-1, weight_avg.size(-1))
        return sel


class Oli_Network_2(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Oli_Network_2, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.classifier = nn.Sequential(
            SepConv_ReLU(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg):
        weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
        sel, state = self.convlstm(weights)
        sel = torch.squeeze(torch.squeeze(torch.stack(sel), dim=0), dim=0)
        sel = self.classifier(sel).view(-1, weight_avg.size(-1))
        return sel


class Oli_Network_3(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Oli_Network_3, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.classifier = nn.Sequential(
            SepConv_ReLU(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg):
        weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
        sel, state = self.convlstm(weights)
        sel = torch.squeeze(torch.squeeze(torch.stack(sel), dim=0), dim=0)
        sel = self.classifier(sel).view(-1, weight_avg.size(-1))
        return sel


class Oli_Network_4(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Oli_Network_4, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.classifier = nn.Sequential(
            SepConv_ReLU(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg):
        weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
        sel, state = self.convlstm(weights)
        sel = torch.squeeze(torch.squeeze(torch.stack(sel), dim=0), dim=0)
        sel = self.classifier(sel).view(-1, weight_avg.size(-1))
        return sel


class Extract(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(Extract, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Conv(x)


class SepConv_Sigmoid(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(SepConv_Sigmoid, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.Conv(x)


class Astro_Network_1(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Astro_Network_1, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.extract1 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.extract2 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.classifier1 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=hidden_size * expansion
            )
        )
        self.classifier2 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )
        self.SepConv = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 5:
            weights = weight_avg.view(1, 1, -1, weight_avg.size(-1))
            weight_avg1 = self.extract1(weights).view(-1, 1, weight_avg.size(-1))
            weight_avg2 = self.extract2(weights).view(-1, 1, weight_avg.size(-1))
            weights = torch.cat((weight_avg1, weight_avg2), dim=1).view(-1, 2, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.cat((torch.squeeze(torch.stack(pr), dim=0)[0], torch.squeeze(torch.stack(pr), dim=0)[-1]), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)

            pr_1 = self.SepConv(pr1)
            pr_1_1 = pr_1[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr_1_2 = pr_1[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            pr2_1 = pr2[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr2_2 = pr2[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            return ((pr_1_1, pr_1_2), (pr2_1, pr2_2))
        else:
            weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.squeeze(torch.squeeze(torch.stack(pr), dim=0), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)
            return pr2


class Astro_Network_2(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Astro_Network_2, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.extract1 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.extract2 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.classifier1 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=hidden_size * expansion
            )
        )
        self.classifier2 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )
        self.SepConv = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 6:
            weights = weight_avg.view(1, 1, -1, weight_avg.size(-1))
            weight_avg1 = self.extract1(weights).view(-1, 1, weight_avg.size(-1))
            weight_avg2 = self.extract2(weights).view(-1, 1, weight_avg.size(-1))
            weights = torch.cat((weight_avg1, weight_avg2), dim=1).view(-1, 2, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.cat((torch.squeeze(torch.stack(pr), dim=0)[0], torch.squeeze(torch.stack(pr), dim=0)[-1]), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)

            pr_1 = self.SepConv(pr1)
            pr_1_1 = pr_1[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr_1_2 = pr_1[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            pr2_1 = pr2[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr2_2 = pr2[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            return ((pr_1_1, pr_1_2), (pr2_1, pr2_2))
        else:
            weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.squeeze(torch.squeeze(torch.stack(pr), dim=0), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)
            return pr2


class Astro_Network_3(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Astro_Network_3, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.extract1 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.extract2 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.classifier1 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=hidden_size * expansion
            )
        )
        self.classifier2 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )
        self.SepConv = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 7:
            weights = weight_avg.view(1, 1, -1, weight_avg.size(-1))
            weight_avg1 = self.extract1(weights).view(-1, 1, weight_avg.size(-1))
            weight_avg2 = self.extract2(weights).view(-1, 1, weight_avg.size(-1))
            weights = torch.cat((weight_avg1, weight_avg2), dim=1).view(-1, 2, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.cat((torch.squeeze(torch.stack(pr), dim=0)[0], torch.squeeze(torch.stack(pr), dim=0)[-1]), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)

            pr_1 = self.SepConv(pr1)
            pr_1_1 = pr_1[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr_1_2 = pr_1[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            pr2_1 = pr2[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr2_2 = pr2[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            return ((pr_1_1, pr_1_2), (pr2_1, pr2_2))
        else:
            weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.squeeze(torch.squeeze(torch.stack(pr), dim=0), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)
            return pr2


class Astro_Network_4(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, kernel_size=(3, 3), num_layers=2, expansion=1):
        super(Astro_Network_4, self).__init__()
        self.convlstm = ConvLSTM(input_size, hidden_size, kernel_size, num_layers)
        self.extract1 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.extract2 = nn.Sequential(
            Extract(
                channel_in=1 * expansion,
                channel_out=1 * expansion
            )
        )
        self.classifier1 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=hidden_size * expansion
            )
        )
        self.classifier2 = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )
        self.SepConv = nn.Sequential(
            SepConv_Sigmoid(
                channel_in=hidden_size * expansion,
                channel_out=1 * expansion
            )
        )

    def forward(self, weight_avg, i):
        if i == 8:
            weights = weight_avg.view(1, 1, -1, weight_avg.size(-1))
            weight_avg1 = self.extract1(weights).view(-1, 1, weight_avg.size(-1))
            weight_avg2 = self.extract2(weights).view(-1, 1, weight_avg.size(-1))
            weights = torch.cat((weight_avg1, weight_avg2), dim=1).view(-1, 2, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.cat((torch.squeeze(torch.stack(pr), dim=0)[0], torch.squeeze(torch.stack(pr), dim=0)[-1]), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)

            pr_1 = self.SepConv(pr1)
            pr_1_1 = pr_1[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr_1_2 = pr_1[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            pr2_1 = pr2[0:weight_avg.size(0)].view(-1, weight_avg.size(-1))
            pr2_2 = pr2[weight_avg.size(0):].view(-1, weight_avg.size(-1))
            return ((pr_1_1, pr_1_2), (pr2_1, pr2_2))
        else:
            weights = weight_avg.view(-1, 1, 1, 1, weight_avg.size(-1))
            pr, state = self.convlstm(weights)
            pr = torch.squeeze(torch.squeeze(torch.stack(pr), dim=0), dim=0)
            pr1 = self.classifier1(pr)
            pr2 = self.classifier2(pr1)
            return pr2