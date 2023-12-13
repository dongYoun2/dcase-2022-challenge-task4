"""
https://github.com/frednam93/FDY-SED
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from desed_task.nnet.CNN import GLU, ContextGating
from desed_task.nnet.RNN import BidirectionalGRU


class Dynamic_conv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        n_basis_kernels=4,
        temperature=31,
        pool_dim="freq",
    ):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(
            in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels, temperature, pool_dim
        )

        self.weight = nn.Parameter(
            torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size), requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x):  # x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ["freq", "chan"]:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)  # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == "time":
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)  # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == "both":
            softmax_attention = (
                self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(
            -1, self.in_planes, self.kernel_size, self.kernel_size
        )  # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ["freq", "chan"]:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == "time":
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output


class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == "both":
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        if self.pool_dim == "freq":
            x = torch.mean(x, dim=3)  # x size : [bs, chan, frames]
        elif self.pool_dim == "time":
            x = torch.mean(x, dim=2)  # x size : [bs, chan, freqs]
        elif self.pool_dim == "both":
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == "chan":
            x = torch.mean(x, dim=1)  # x size : [bs, freqs, frames]

        if not self.pool_dim == "both":
            x = self.conv1d1(x)  # x size : [bs, hid_chan, frames | freqs]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)  # x size : [bs, n_ker, frames | freqs]
        else:
            x = self.fc1(x)  # x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)  # x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)


class FDCNN(nn.Module):  # Frequency Dynamic Convolution
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0.5,
        kernel_size=[3, 3, 3, 3, 3, 3, 3],
        padding=[1, 1, 1, 1, 1, 1, 1],
        stride=[1, 1, 1, 1, 1, 1, 1],
        nb_filters=[32, 64, 128, 256, 256, 256, 256],
        pooling=[[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
        normalization="batch",
        n_basis_kernels=4,
        DY_layers=[0, 1, 1, 1, 1, 1, 1],
        temperature=31,
        pool_dim="time",
        **kwargs,
    ):
        super(FDCNN, self).__init__()
        self.nb_filters = nb_filters
        self.n_filt_last = self.nb_filters[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            in_dim = n_in_channel if i == 0 else self.nb_filters[i - 1]
            out_dim = self.nb_filters[i]
            if DY_layers[i] == 1:
                cnn.add_module(
                    "conv{0}".format(i),
                    Dynamic_conv2d(
                        in_dim,
                        out_dim,
                        kernel_size[i],
                        stride[i],
                        padding[i],
                        n_basis_kernels=n_basis_kernels,
                        temperature=temperature,
                        pool_dim=pool_dim,
                    ),
                )
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel_size[i], stride[i], padding[i]))
            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(self.nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):  # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class FDCRNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        nclass=10,
        activation="cg",
        conv_dropout=0.5,
        n_RNN_cell=256,
        rnn_layers=2,
        rec_dropout=0,
        attention=True,
        T=1,
        **convkwargs,
    ):
        super(FDCRNN, self).__init__()
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.nclass = nclass
        self.T = T

        self.cnn = FDCNN(n_in_channel=self.n_in_channel, activation=activation, conv_dropout=conv_dropout, **convkwargs)
        self.rnn = BidirectionalGRU(
            n_in=self.cnn.nb_filters[-1], n_hidden=n_RNN_cell, dropout=rec_dropout, num_layers=rnn_layers
        )

        self.dropout = nn.Dropout(conv_dropout)
        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Linear(n_RNN_cell * 2, self.nclass)

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, self.nclass)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # input size : [bs, freqs, frames]
        # cnn
        if self.n_in_channel > 1:
            x = x.transpose(2, 3)
        else:
            x = x.transpose(1, 2).unsqueeze(1)  # x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        bs, ch, frame, freq = x.size()
        if freq != 1:
            print("warning! frequency axis is large: " + str(freq))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous.view(bs, frame, ch * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # x size : [bs, frames, chan]

        # rnn
        x = self.rnn(x)  # x size : [bs, frames, 2 * chan]
        x = self.dropout(x)

        # classifier
        strong = self.dense(x)  # strong size : [bs, frames, nclass]
        strong = self.sigmoid(strong / self.T)
        if self.attention:
            sof = self.dense_softmax(x)  # sof size : [bs, frames, nclass]
            sof = self.softmax(sof)  # sof size : [bs, frames, nclass]
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)

        return strong.transpose(1, 2), weak


if __name__ == "__main__":
    model = FDCRNN(1)

    output = model(torch.rand(4, 128, 512))
