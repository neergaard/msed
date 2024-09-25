from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# from torch.autograd import Variable

from msed.models.base import BaseNet


# fmt: off
class DOSED(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        self.spatial_filter = nn.Conv2d(1, self.n_channels, (self.n_channels, 1))
        self.blocks = nn.ModuleList([
            nn.Sequential(OrderedDict([
                (f'conv_{k}', nn.Conv2d(in_channels=self.filter_base * 2 ** (k - 1) if k > 1 else 1,
                                        out_channels=self.filter_base * 2 ** k,
                                        kernel_size=(1, self.kernel_size),
                                        stride=1,
                                        padding=(0, self.padding))),
                (f'batchnorm_{k}', nn.BatchNorm2d(self.filter_base * 2 ** k)),
                (f'relu_{k}', nn.ReLU()),
                (f'maxpool_{k}', nn.MaxPool2d(kernel_size=(1, 2))),
            ])) for k in range(1, self.k_max + 1)
        ])

        self.localization = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * (2 ** self.k_max),
                out_channels=2 * len(self.localizations_default),
                kernel_size=(self.n_channels, int(self.window_size / (2 ** self.k_max))),
                padding=0),
            Rearrange('B (F f) C T -> B F (f C T)', f=2),
        )
        self.classification = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * (2 ** self.k_max),
                out_channels=self.n_classes * len(self.localizations_default),
                kernel_size=(self.n_channels, int(self.window_size / (2 ** self.k_max))),
                padding=0),
            Rearrange('B (F f) C T -> B F (f C T)', f=self.n_classes),
        )

        self.localizations_default_expanded = torch.tensor(self.localizations_default).to(self.device)
        # self.register_buffer('localizations_default_expanded', torch.tensor(self.localizations_default))

    def forward(self, x):

        N, C, T = x.shape
        z = rearrange(x, 'N C T -> N 1 C T')

        z = self.spatial_filter(z)
        z = rearrange(z, 'N C 1 T -> N 1 C T')

        for block in self.blocks:
            z = block(z)

        loc = self.localization(z)
        torch.cuda.empty_cache()
        clf = self.classification(z)
        torch.cuda.empty_cache()
        # print(loc.shape)
        # print(clf.shape)
        # print(x.shape)
        return loc, clf


class RnnModel(BaseNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)
        self.spatial_filtering = nn.Conv2d(1, self.n_channels, (self.n_channels, 1))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    (f"conv_{k}", nn.Conv2d(in_channels=self.filter_base * 2 ** (k - 1) if k > 1 else self.n_channels,
                                            out_channels=self.filter_base * 2 ** k,
                                            kernel_size=(1, self.kernel_size),
                                            stride=(1, 2),
                                            padding=(0, self.padding),
                                            bias=False)),
                    (f"batchnorm_{k}", nn.BatchNorm2d(self.filter_base * 2 ** k)),
                    (f"relu_{k}", nn.ReLU())
                ])
            ) for k in range(1, self.k_max + 1)])

        self.recurrent = nn.GRU(input_size=self.filter_base * 2 ** self.k_max,
                                hidden_size=self.recurrent_n_hidden,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        # self.localization = nn.Sequential(
        #     OrderedDict([
        #         ('loc_1', nn.Conv2d(in_channels=self.filter_base * (2 ** self.k_max),
        #                             out_channels=self.filter_base * (2 ** self.k_max) // 2,
        #                             kernel_size=(2, int(self.window_size / (2 ** self.k_max))))),
        #         ('loc_2', nn.Conv2d(in_channels=self.filter_base * (2 ** self.k_max) // 2,
        #                             out_channels=2 * len(self.localizations_default),
        #                             kernel_size=(1, 1)))
        #     ])
        # )
        # self.localizations = nn.Sequential(
        #     nn.Conv2d(in_channels=self.filter_base * (2 ** self.k_max),
        #               out_channels=self.filter_base * (2 ** self.k_max),
        #               kernel_size=(2, int(self.window_size / (2 ** self.k_max)))),
        # nn.Conv2d(in_channels=self.filter_base * (2 ** self.k_max),
        #           out_channels=2 * len(self.localizations_default),
        #           kernel_size=(1, 1))
        # )
        self.localization = nn.Conv2d(in_channels=self.recurrent_n_hidden,
                                      out_channels=2 * len(self.localizations_default),
                                      kernel_size=(2, int(self.window_size / (2 ** self.k_max))))

        self.classification = nn.Conv2d(in_channels=self.recurrent_n_hidden,
                                        out_channels=self.n_classes *
                                        len(self.localizations_default),
                                        kernel_size=(2, int(self.window_size / (2 ** self.k_max))))

        self.localizations_default_expanded = self.localizations_default
        # self.localization = nn.Conv2d(in_channels=self.filter_base * 2 ** self.k_max,
        #                               out_channels=2 * len(self.localizations_default),
        #                               kernel_size=(2, 1))

        # self.classification = nn.Conv2d(in_channels=self.filter_base * 2 ** self.k_max,
        #                                 out_channels=self.n_classes * len(self.localizations_default),
        #                                 kernel_size=(2, 1))

        # self.localizations_default_expanded = np.tile(self.localizations_default[None, :], [
        #                                               self.window_size // self.fs, 1, 1]).reshape(-1, 2)

    def forward(self, x):
        self.recurrent.flatten_parameters()
        batch = x.size(0)
        size = x.size()
        z = x.view(size[0], 1, size[1], size[2])

        if self.n_channels != 1:
            z = self.spatial_filtering(z)

        for block in self.blocks:
            z = block(z)

        zdim = z.size()
        window_size = zdim[-1]
        z = z.permute(0, 3, 2, 1).reshape(batch, window_size, -1)
        z = self.recurrent(z)[0].view(batch, window_size, 2, -1).permute(0, 3, 2, 1)

        # Possibly dropout
        if hasattr(self, 'dropout_layer'):
            z = self.dropout_layer(z)

        # Possible dropout
        # if self.dropout:
        #     z = self.dropout_layer(z)

        loc = self.localization(z).squeeze().view(batch, -1, 2)
        clf = self.classification(z).squeeze().view(batch, -1, self.n_classes)
        # return loc, clf, self.localizations_default
        # loc = self.localization(z).squeeze(2).permute(0, 2, 1).reshape(batch, window_size, -1, 2)
        # clf = self.classification(z).squeeze(2).permute(0, 2, 1).reshape(batch, window_size, -1, self.n_classes)
        # loc = loc.reshape(batch, -1, 2)
        # clf = clf.reshape(batch, -1, self.n_classes)

#         if torch.cuda.device_count() > 1:
#             print(self.window_size // self.fs)
#             localizations_default = self.localizations_default.unsqueeze(0).repeat([self.window_size // self.fs, 1, 1]).reshape(-1, 2)
#         else:
#             print(self.window_size // self.fs)
#             localizations_default = self.localizations_default.unsqueeze(0).repeat([self.window_size // self.fs, 1, 1]).reshape(-1, 2)

        return loc, clf


class ResidualBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, kernel_size):
        super().__init__()
        self.filters_in = n_filters_in
        self.filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(self.filters_in),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters_in,
                      out_channels=self.filters_out,
                      kernel_size=(1, self.kernel_size),
                      stride=(1, 2),
                      padding=(0, self.padding),
                      bias=False),
            nn.BatchNorm2d(self.filters_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters_out,
                      out_channels=self.filters_out,
                      kernel_size=(1, self.kernel_size),
                      padding=(0, self.padding),
                      bias=False))
        self.projection = nn.Conv2d(in_channels=self.filters_in,
                                    out_channels=self.filters_out,
                                    kernel_size=(1, 1),
                                    stride=(1, 2),
                                    bias=False)

    def forward(self, x):
        shortcut = x
        x = self.net(x)
        # print(x.shape)
        return self.projection(shortcut) + x


class BottleneckBlock(nn.Module):

    def __init__(self, n_filters_in, n_filters_out, kernel_size, strided=False):
        super().__init__()
        self.filters_in = n_filters_in
        self.filters_out = n_filters_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = (1, 2) if strided else (1, 1)
        self.net = nn.Sequential(
            nn.BatchNorm2d(self.filters_in),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters_in,
                      out_channels=self.filters_out // 4,
                      kernel_size=(1, 1),
                      #   stride=self.stride,
                      #   padding=(0, self.padding),
                      bias=False),
            nn.BatchNorm2d(self.filters_out // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters_out // 4,
                      out_channels=self.filters_out // 4,
                      kernel_size=(1, self.kernel_size),
                      padding=(0, self.padding),
                      stride=self.stride,
                      bias=False),
            nn.BatchNorm2d(self.filters_out // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.filters_out // 4,
                      out_channels=self.filters_out,
                      kernel_size=(1, 1),
                      #   padding=(0, self.padding),
                      bias=False))
        self.projection = nn.Conv2d(in_channels=self.filters_in,
                                    out_channels=self.filters_out,
                                    kernel_size=self.stride,
                                    stride=self.stride,
                                    bias=False)

    def forward(self, x):
        shortcut = x
        x = self.net(x)
        # print(x.shape)
        return self.projection(shortcut) + x


class ResNetRnn(BaseNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        # Architecture components
        self.spatial_filtering = nn.Conv2d(
            1, self.n_channels, (self.n_channels, 2), stride=(1, 2), bias=False)

        residual_blocks = []
        for k in range(1, self.k_max + 1):
            n_filters_out = self.filter_base * 2 ** (k - 1)
            for r in range(self.n_repeats):
                if k == 1 and r == 0:
                    n_filters_in = self.n_channels
                elif r == 0:
                    n_filters_in = self.filter_base * 2 ** (k - 2)
                else:
                    n_filters_in = self.filter_base * 2 ** (k - 1)
                residual_blocks.append(
                    BottleneckBlock(n_filters_in=n_filters_in,
                                    n_filters_out=n_filters_out,
                                    kernel_size=self.kernel_size,
                                    strided=True if r == 0 and k > 1 else False)
                )
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.recurrent = nn.GRU(input_size=self.filter_base * 2 ** (self.k_max - 1),
                                hidden_size=self.recurrent_n_hidden,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.localization = nn.Conv2d(in_channels=self.recurrent_n_hidden,
                                      out_channels=2 * len(self.localizations_default),
                                      kernel_size=(2, int(self.window_size / (2 ** self.k_max))))

        self.classification = nn.Conv2d(in_channels=self.recurrent_n_hidden,
                                        out_channels=self.n_classes *
                                        len(self.localizations_default),
                                        kernel_size=(2, int(self.window_size / (2 ** self.k_max))))

        self.localizations_default_expanded = self.localizations_default

    def forward(self, x):

        batch = x.size(0)
        size = x.size()
        z = x.view(size[0], 1, size[1], size[2])

        if self.n_channels != 1:
            z = self.spatial_filtering(z)

        for residual_block in self.residual_blocks:
            z = residual_block(z)

        zdim = z.size()
        window_size = zdim[-1]
        z = z.permute(0, 3, 2, 1).reshape(batch, window_size, -1)
        z = self.recurrent(z)[0].view(batch, window_size, 2, -1).permute(0, 3, 2, 1)

        # Possibly dropout
        if hasattr(self, 'dropout_layer'):
            z = self.dropout_layer(z)

        loc = self.localization(z).squeeze().view(batch, -1, 2)
        clf = self.classification(z).squeeze().view(batch, -1, self.n_classes)

        return loc, clf

    # def __init__(self, **kwargs):
    #     (super().__init__)(**kwargs)
    #     self.__dict__.update(kwargs)
    #     self.spatial_filtering = nn.Conv2d(1, self.n_channels, (self.n_channels, 1), bias=False)
    #     self.residual_blocks = nn.ModuleList([
    #         BottleneckBlock(n_filters_in=self.n_channels if k == 1 else self.filter_base * 2 ** (k - 1),
    #                         n_filters_out=self.filter_base * 2 ** k,
    #                         kernel_size=self.kernel_size
    #         ) for k in range(1, self.k_max + 1)])
    #     self.recurrent = nn.GRU(input_size=(self.filter_base * 2 ** self.k_max),
    #                             hidden_size=(self.filter_base * 2 ** self.k_max),
    #                             num_layers=1,
    #                             batch_first=True,
    #                             dropout=0,
    #                             bidirectional=True)
    #     self.localization = nn.Conv2d(in_channels=self.filter_base * 2 ** self.k_max,
    #                                   out_channels=2 * len(self.localizations_default),
    #                                   kernel_size=(2, 1))
    #     self.classification = nn.Conv2d(in_channels=self.filter_base * 2 ** self.k_max,
    #                                     out_channels=self.n_classes * len(self.localizations_default),
    #                                     kernel_size=(2, 1))
    #     # self.map2device()
    #     self.localizations_default_expanded = np.tile(self.localizations_default[None, :], [
    #                                                   self.window_size // self.fs, 1, 1]).reshape(-1, 2)

    # def forward(self, x):
    #     self.recurrent.flatten_parameters()
    #     batch = x.size(0)
    #     size = x.size()
    #     z = x.view(size[0], 1, size[1], size[2])

    #     if self.n_channels != 1:
    #         z = self.spatial_filtering(z)

    #     for residual_block in self.residual_blocks:
    #         z = residual_block(z)

    #     zdim = z.size()
    #     window_size = zdim[(-1)]
    #     z = z.permute(0, 3, 2, 1).reshape(batch, window_size, -1)
    #     z = self.recurrent(z)[0].view(batch, window_size, 2, -1).permute(0, 3, 2, 1)

    #     loc = self.localization(z).squeeze(2).permute(0, 2, 1).reshape(batch, window_size, -1, 2)
    #     clf = self.classification(z).squeeze(2).permute(0, 2, 1).reshape(batch, window_size, -1, self.n_classes)
    #     loc = loc.reshape(batch, -1, 2)
    #     clf = clf.reshape(batch, -1, self.n_classes)
    #     return loc, clf
        # localizations_default = self.localizations_default.unsqueeze(0).repeat([window_size, 1, 1]).reshape(-1, 2)
        # return loc, clf, localizations_default


class ResNet(BaseNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        # Architecture components
        self.spatial_filtering = nn.Conv2d(
            1, self.n_channels, (self.n_channels, 2), stride=(1, 2), bias=False)

        residual_blocks = []
        for k in range(1, self.k_max + 1):
            n_filters_out = self.filter_base * 2 ** (k - 1)
            for r in range(self.n_repeats):
                if k == 1 and r == 0:
                    n_filters_in = self.n_channels
                elif r == 0:
                    n_filters_in = self.filter_base * 2 ** (k - 2)
                else:
                    n_filters_in = self.filter_base * 2 ** (k - 1)
                residual_blocks.append(
                    BottleneckBlock(n_filters_in=n_filters_in,
                                    n_filters_out=n_filters_out,
                                    kernel_size=self.kernel_size,
                                    strided=True if r == 0 and k > 1 else False)
                )
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.localization = nn.Conv2d(in_channels=self.filter_base * 2 ** (self.k_max - 1),
                                      out_channels=2 * len(self.localizations_default),
                                      kernel_size=(1, int(self.window_size / (2 ** self.k_max))))

        self.classification = nn.Conv2d(in_channels=self.filter_base * 2 ** (self.k_max - 1),
                                        out_channels=self.n_classes *
                                        len(self.localizations_default),
                                        kernel_size=(1, int(self.window_size / (2 ** self.k_max))))

        self.localizations_default_expanded = self.localizations_default

    def forward(self, x):

        batch = x.size(0)
        size = x.size()
        z = x.view(size[0], 1, size[1], size[2])

        if self.n_channels != 1:
            z = self.spatial_filtering(z)

        for residual_block in self.residual_blocks:
            z = residual_block(z)

        loc = self.localization(z).squeeze().view(batch, -1, 2)
        clf = self.classification(z).squeeze().view(batch, -1, self.n_classes)

        return loc, clf


class ResNetRnnAtt(ResNetRnn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        self.attention = AdditiveAttntn(input_size=2 * self.recurrent_n_hidden,
                                        hidden_size=self.attention_n_hidden)

        self.localization = nn.Conv1d(in_channels=2 * self.recurrent_n_hidden,
                                      out_channels=2 * len(self.localizations_default),
                                      kernel_size=1)

        self.classification = nn.Conv1d(in_channels=2 * self.recurrent_n_hidden,
                                        out_channels=self.n_classes *
                                        len(self.localizations_default),
                                        kernel_size=1)

    def forward(self, x):
        self.recurrent.flatten_parameters()

        batch = x.size(0)
        size = x.size()
        z = x.view(size[0], 1, size[1], size[2])

        if self.n_channels != 1:
            z = self.spatial_filtering(z)

        for residual_block in self.residual_blocks:
            z = residual_block(z)

        zdim = z.size()
        window_size = zdim[-1]
        z = z.permute(0, 3, 2, 1).reshape(batch, window_size, -1)
        z = self.recurrent(z)[0].view(batch, window_size, -1)  # , 2, -1).permute(0, 3, 2, 1)

        # Attention layer
        z = self.attention(z).unsqueeze(-1)

        # Possibly dropout
        if hasattr(self, 'dropout_layer'):
            z = self.dropout_layer(z)

        loc = self.localization(z).squeeze().view(batch, -1, 2)
        clf = self.classification(z).squeeze().view(batch, -1, self.n_classes)

        return loc, clf

# Additive Attention


class AdditiveAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.compute_u = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Tanh()
        )
        self.compute_a = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, h):
        # h.size() = [Batch size, Input size, Sequence length]
        h = h.permute(0, 2, 1)
        u = self.compute_u(h)
        a = self.compute_a(u).permute(0, 2, 1)
        m = torch.matmul(a, h)

        return m, a


class Stream(nn.Module):
    def __init__(self, filter_base, k_max, kernel_size, n_channels, name, padding):
        super().__init__()

        self.filter_base = filter_base
        self.k_max = k_max
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.name = name
        self.padding = padding

        self.spatial = nn.Sequential(
            nn.Conv2d(1, self.n_channels, (self.n_channels, 1), bias=False),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                OrderedDict([
                    (f"{name}:conv_{k}", nn.Conv2d(in_channels=self.filter_base * 2 ** (k - 1) if k > 1 else self.n_channels,
                                                   out_channels=self.filter_base * 2 ** k,
                                                   kernel_size=(1, self.kernel_size),
                                                   stride=(1, 2),
                                                   padding=(0, self.padding),
                                                   bias=False)),
                    (f"{name}batchnorm_{k}", nn.BatchNorm2d(self.filter_base * 2 ** k)),
                    (f"{name}relu_{k}", nn.ReLU())
                ])
            ) for k in range(1, self.k_max + 1)])

    def forward(self, x):
        """Input has dimensions [Batch size, n_channels, time]"""
        z = x.unsqueeze(1)

        if self.n_channels != 1:
            z = self.spatial(z)

        for block in self.blocks:
            z = block(z)

        return z

class ResidualStream(Stream):
    def __init__(self, filter_base, k_max, kernel_size, n_channels, name, padding, n_repeats):
        super().__init__(filter_base, k_max, kernel_size, n_channels, name, padding)

        self.n_repeats = n_repeats

        residual_blocks = []
        for k in range(1, self.k_max + 1):
            n_filters_out = self.filter_base * 2 ** k
            for r in range(self.n_repeats):
                if k == 1 and r == 0:
                    n_filters_in = self.n_channels
                elif r == 0:
                    n_filters_in = self.filter_base * 2 ** (k - 1)
                else:
                    n_filters_in = self.filter_base * 2 ** k
                residual_blocks.append(
                    BottleneckBlock(n_filters_in=n_filters_in,
                                    n_filters_out=n_filters_out,
                                    kernel_size=self.kernel_size,
                                    strided=True if r == 0 and k > 1 else False)
                )
        self.blocks = nn.Sequential(*residual_blocks)

    def forward(self, x):
        z = super().forward(x)

        return z


class SplitStreamNet(BaseNet):
    """The Split-Stream network architecture disentangles the initial feature extraction for each major sleep event type
    by utilizing a separate fully convolutional architecture for each event. The features for each event are concanted downstream
    and processed temporally using a recurrent neural network.
        The network also uses a self-attention mechanism to the merged feature vectors and depth-wise separable convolutions
    to classify and localize the sleep events.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        # Define streams
        self.arousal_stream = Stream(self.filter_base, self.k_max,
                                     self.kernel_size, 5, 'arousal', self.padding)
        self.lm_stream = Stream(self.filter_base, self.k_max,
                                self.kernel_size, 2, 'lm', self.padding)
        self.sdb_stream = Stream(self.filter_base, self.k_max,
                                 self.kernel_size, 3, 'sdb', self.padding)

        # Define layers after concatenating streams
        self.recurrent = nn.GRU(input_size=3 * self.filter_base * 2 ** self.k_max,
                                hidden_size=self.recurrent_n_hidden,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)
        self.attention = AdditiveAttention(input_size=2 * self.recurrent_n_hidden,
                                           hidden_size=self.attention_n_hidden,
                                           output_size=self.n_classes)
        self.localization = nn.Conv1d(in_channels=self.n_classes,
                                      out_channels=2 * len(self.localizations_default),
                                      kernel_size=self.recurrent_n_hidden * 2,
                                      groups=self.n_classes if self.depthwise == "True" or self.depthwise == True else 1)
        self.classification = nn.Conv1d(in_channels=self.n_classes,
                                        out_channels=self.n_classes *
                                        len(self.localizations_default),
                                        kernel_size=self.recurrent_n_hidden * 2,
                                        groups=self.n_classes if self.depthwise == "True" or self.depthwise == True else 1)

        self.localizations_default_expanded = self.localizations_default

    def forward(self, x):
        """Input dimension is [N, n_channels, duration_min * sampling_rate]"""
        self.recurrent.flatten_parameters()
        N, C, T = x.shape
        x_ar, x_lm, x_sdb = torch.split(x, [5, 2, 3], dim=1)

        z_ar = self.arousal_stream(x_ar)
        z_lm = self.lm_stream(x_lm)
        z_sdb = self.sdb_stream(x_sdb)

        # Concatenate and reshape to [N, steps, features] for the recurrent layer
        z = torch.cat([z_ar, z_lm, z_sdb], dim=1).squeeze(2).permute(0, 2, 1)
        z = self.recurrent(z)[0].permute(0, 2, 1)
        z, a = self.attention(z)

        # Do detection here
        clf = self.classification(z).squeeze().reshape(N, -1, self.n_classes)
        loc = self.localization(z).squeeze().reshape(N, -1, 2)

        return loc, clf


class ResidualSplitStreamNet(SplitStreamNet):
    """The Residual-Split-Stream network architecture disentangles the initial feature extraction for each major sleep event type
    by utilizing a separate fully convolutional architecture for each event. The features for each event are concanted downstream
    and processed temporally using a recurrent neural network.
        The network also uses a self-attention mechanism to the merged feature vectors and depth-wise separable convolutions
    to classify and localize the sleep events.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        # Define streams
        self.arousal_stream = ResidualStream(self.filter_base, self.k_max,
                                     self.kernel_size, 5, 'arousal', self.padding, self.n_repeats)
        self.lm_stream = ResidualStream(self.filter_base, self.k_max,
                                self.kernel_size, 2, 'lm', self.padding, self.n_repeats)
        self.sdb_stream = ResidualStream(self.filter_base, self.k_max,
                                 self.kernel_size, 3, 'sdb', self.padding, self.n_repeats)

    def forward(self, x):
        """Input dimension is [N, n_channels, duration_min * sampling_rate]"""
        self.recurrent.flatten_parameters()
        N, C, T = x.shape
        x_ar, x_lm, x_sdb = torch.split(x, [5, 2, 3], dim=1)

        z_ar = self.arousal_stream(x_ar)
        z_lm = self.lm_stream(x_lm)
        z_sdb = self.sdb_stream(x_sdb)

        # Concatenate and reshape to [N, steps, features] for the recurrent layer
        z = torch.cat([z_ar, z_lm, z_sdb], dim=1).squeeze(2).permute(0, 2, 1)
        z = self.recurrent(z)[0].permute(0, 2, 1)
        z, a = self.attention(z)

        # Do detection here
        clf = self.classification(z).squeeze().reshape(N, -1, self.n_classes)
        loc = self.localization(z).squeeze().reshape(N, -1, 2)

        return loc, clf


if __name__ == '__main__':

    import json

    json_str = '{"default_event_sizes": [3, 15, 30], "detection_parameters": {"classification_threshold": 0.7, "overlap_non_maximum_suppression": 0.5, "softmax": true}, "device": "cuda", "factor_overlap": 2, "filter_base": 4, "fs": 128, "k_max": 7, "maxpool_kernel_size": 2, "kernel_size": 3, "pdrop": 0.1, "recurrent_n_hidden": 128}'
    network_params = json.loads(json_str)

    def test_attention():
        net = AdditiveAttention(256, 64, 3)
        x = torch.randn(32, 256, 120)

        m, a = net(x)
        print('m.shape:', m.shape)
        print('a.shape:', a.shape)

    def test_stream():
        net = Stream(4, 7, 3, 5, 'arousal', 3 // 2)
        x = torch.randn(32, 5, 120*128)

        z = net(x)
        print('z.shape:', z.shape)

    def test_SplitStreamNet():
        new_params = {'n_channels': 10, 'n_classes': 4, 'window_size': 128 * 120,
                      'dropout': None, 'attention_n_hidden': 128, 'recurrent_n_hidden': 128,
                      'filter_base': 4, 'depthwise': False}
        network_params.update(new_params)
        print(json.dumps(network_params, indent=4, sort_keys=True))
        net = SplitStreamNet(**network_params)
        print(net)

        x = torch.randn(128, 10, 128 * 120)
        loc, clf = net(x)
        print('clf.shape', clf.shape)
        print('loc.shape', loc.shape)
        net.summary(input_size=(10, 128 * 120), batch_size=128)

    def test_ResidualSplitStreamNet():
        new_params = {'n_channels': 10, 'n_classes': 4, 'window_size': 128 * 120,
                      'dropout': None, 'attention_n_hidden': 128, 'recurrent_n_hidden': 128,
                      'filter_base': 2, 'depthwise': False, 'n_repeats': 2}
        network_params.update(new_params)
        print(json.dumps(network_params, indent=4, sort_keys=True))
        net = ResidualSplitStreamNet(**network_params)
        print(net)

        x = torch.randn(128, 10, 128 * 120)
        loc, clf = net(x)
        print('clf.shape', clf.shape)
        print('loc.shape', loc.shape)
        net.summary(input_size=(10, 128 * 120), batch_size=128)

    def test_RnnModel():
        new_params = {'n_channels': 10, 'n_classes': 4, 'window_size': 128 * 120,
                      'dropout': None, 'recurrent_n_hidden': 128, 'filter_base': 4}
        network_params.update(new_params)
        print(json.dumps(network_params, indent=4, sort_keys=True))
        net = RnnModel(**network_params)
        print(net)

        x = torch.randn(32, 10, 128 * 120)
        loc, clf = net(x)
        print('clf.shape', clf.shape)
        print('loc.shape', loc.shape)
        net.summary(input_size=(10, 128 * 120), batch_size=32)

    # import json

    # json_str = '{"default_event_sizes": [3, 15, 30], "detection_parameters": {"classification_threshold": 0.7, "overlap_non_maximum_suppression": 0.5, "softmax": true}, "device": "cuda", "factor_overlap": 2, "filter_base": 4, "fs": 128, "k_max": 7, "maxpool_kernel_size": 2, "kernel_size": 3, "pdrop": 0.1, "recurrent_n_hidden": 128}'
    # network_params = json.loads(json_str)
    # additional_network_params = {'n_channels': 10,
    #                              'n_classes': 4,
    #                              'window_size': 128 * 120,
    #                              'dropout': None,
    #                              'n_repeats': 3,
    #                              'attention_n_hidden': 512}
    # network_params.update(additional_network_params)
    # print(network_params)
    # model = ResNetRnn(**network_params)
    # # print(model)
    # model.summary(input_size=(10, 120*128), batch_size=32)

    # test_attention()
    # test_stream()
    test_SplitStreamNet()
    # test_ResidualSplitStreamNet()
    # test_RnnModel()
# fmt: off
