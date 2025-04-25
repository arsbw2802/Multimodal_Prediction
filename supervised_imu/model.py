import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch import nn
from typing import List


class Encoder(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        embedding_dim: int,
        encoder_type: str,
        additional_params: dict = {},
    ) -> None:
        super().__init__()

        if encoder_type == "conv":
            self.model = HarishConvEncoder(num_of_features)

        elif encoder_type == "lstm":
            self.model = LstmEncoder(num_of_features, embedding_dim)
        elif encoder_type == "deepconvlstm":
            self.model = DeepConvLstmEncoder(num_of_features, embedding_dim)
        elif encoder_type == "cpc":
            self.model = CPCEncoder(
                num_of_features,
                embedding_dim,
                additional_params["num_steps_prediction"],
            )
        elif encoder_type=="resnet":
            self.model =ResNet(num_of_features, additional_params["resnet_type"])
        elif encoder_type=="abrar":
            self.model =AbrarResNet()
        

    def forward(self, inputs):
        return self.model(inputs)

    def predict_features(self, inputs):
        return self.model.predict_features(inputs)

class Classifier(nn.Module):  ###classifier head for cross entropy loss
    def __init__(self, embedding_dim, num_of_classes, encoder_type, ln1=256, p=0.2):
        super(Classifier, self).__init__()
        # Defining the two layer MLP
        # print("Embedding dimension", embedding_dim)
        ln2 = ln1 // 2
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.softmax = nn.Sequential(
            nn.Linear(embedding_dim, ln1),
            nn.BatchNorm1d(ln1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(ln1, ln2),
            nn.BatchNorm1d(ln2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(ln2, num_of_classes),
        )

        def _weights_init(m):
            if isinstance(m, nn.Conv1d or nn.Linear or nn.LSTM):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, encoding):
    
        if self.encoder_type == "conv":
            encoding = F.max_pool1d(encoding, kernel_size=encoding.shape[2]).squeeze(2)
        elif self.encoder_type == "lstm":
            encoding = encoding[:, -1, :]
            encoding = encoding.contiguous().view(-1, self.embedding_dim)
        elif self.encoder_type == "deepconvlstm":
            encoding = encoding[:, -1, :]
            encoding = encoding.contiguous().view(-1, self.embedding_dim)
        elif self.encoder_type == "cpc":
            encoding = encoding[:, -1, :]
            encoding = encoding.contiguous().view(-1, self.embedding_dim)
        elif self.encoder_type =="resnet":
            encoding = F.max_pool1d(encoding, kernel_size=encoding.shape[2]).squeeze(2)
        
        out = self.softmax(encoding)

        return out

class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


# Taken from the 700k person days of data paper
class ResNet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
            self,
            num_of_features,
            resnet_type,
    ):
        super(ResNet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)

        # TODO: play around with the architecture here
        if resnet_type == 'resnet_1_block_conv_5':
            cgf = [
            (64, 5, 1, 5, 1, 0),
            (128, 5, 1, 5, 1, 0),
            (256, 5, 1, 5, 1, 0),
            (512, 5, 1, 5, 1, 0),
            ]
        if resnet_type == 'resnet_1_block_conv_3':
            cgf = [
            (64, 3, 1, 3, 1, 0),
            (128, 3, 1, 3, 1, 0),
            (256, 3, 1, 3, 1, 0),
            (512, 3, 1, 3, 1, 0),
            ]

        in_channels = num_of_features
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                ResNet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))

        if downfactor != 1:
            modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(1, 2)
        feats = self.feature_extractor(x)
        return feats


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def getBNones(size):
    bn = nn.BatchNorm1d(size)
    bn.weight.data.fill_(1.0)
    return bn


class AbrarResNet(nn.Module):
    def __init__(self):
        super(AbrarResNet, self).__init__()
        self.relu = nn.ReLU(True)
        self.pool4 = nn.MaxPool1d(12)

        self.conv0 = nn.Conv1d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn0 = getBNones(64)

        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = getBNones(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = getBNones(256)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = getBNones(512)

        self.res1a = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1a = getBNones(128)
        self.res1b = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1b = getBNones(128)

        self.res3a = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3a = getBNones(512)
        self.res3b = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3b = getBNones(512)

        self.flatten = nn.Flatten()

    def forward(self, features):
        features = features.permute(0, 2, 1)
        x = self.conv0(features)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool2(x)  #

        res1 = self.res1a(x)
        res1 = self.bn1a(res1)
        res1 = self.relu(res1)
        res1 = self.res1b(res1)
        res1 = self.bn1b(res1)
        res1 = self.relu(res1)
        a = x + res1

        a = self.conv2(a)
        a = self.bn2(a)
        a = self.relu(a)
        a = self.pool2(a)

        a = self.conv3(a)
        a = self.bn3(a)
        a = self.relu(a)
        a = self.pool2(a)  #

        b = self.res3a(a)
        b = self.bn3a(b)
        b = self.relu(b)
        b = self.res3b(b)
        b = self.bn3b(b)
        b = self.relu(b)
        c = a + b

        c = self.pool4(c)
        c = self.flatten(c)
        # c = self.lin(c)
        # outs = torch.mul(c, 0.125)

        return c

class ConvEncoder(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        # changing conv encoder according to Harish's best performing architecture
        stride = 1
        self.conv1 = nn.Conv1d(
            in_channels=num_of_features,
            out_channels=32,
            kernel_size=24,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=16,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=embedding_dim,
            kernel_size=8,
            stride=stride,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        def _weights_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):

        # input shape to conv1d is (N, C, L) N = Batch Size, C=Input channels, L= Length of Sequence
        # Multiple input channels dont imply adding another dims(does not make the network conv2d).
        # Mulitple channels mean different inputs in the same space. Like for images, we have Red, Green and Blue values for the same position
        # Similarly for signal data, we have 6 different values for the same time value

        inputs = inputs.permute(0, 2, 1)
        conv1_out = self.dropout(self.relu(self.conv1(inputs)))
        conv2_out = self.dropout(self.relu(self.conv2(conv1_out)))
        conv3_out = self.dropout(self.relu(self.conv3(conv2_out)))
        # conv3_out = conv3_out.permute(0, 2, 1)

        return conv3_out

    def predict_features(self, inputs):
        pass
    
    
class ConvEncoderOrg(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        # changing conv encoder according to Harish's best performing architecture
        stride = 1
        kernel_size=3
        self.conv1 = nn.Conv1d(
            in_channels=num_of_features,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        def _weights_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):

        # input shape to conv1d is (N, C, L) N = Batch Size, C=Input channels, L= Length of Sequence
        # Multiple input channels dont imply adding another dims(does not make the network conv2d).
        # Mulitple channels mean different inputs in the same space. Like for images, we have Red, Green and Blue values for the same position
        # Similarly for signal data, we have 6 different values for the same time value

        inputs = inputs.permute(0, 2, 1)
        conv1_out = self.dropout(self.relu(self.conv1(inputs)))
        conv2_out = self.dropout(self.relu(self.conv2(conv1_out)))
        conv3_out = self.dropout(self.relu(self.conv3(conv2_out)))
        # conv3_out = conv3_out.permute(0, 2, 1)

        return conv3_out

    def predict_features(self, inputs):
        pass

class Net(nn.Module):
    def __init__(self, list_of_modules: List) -> None:
        super().__init__()
        self.model = nn.Sequential(*list_of_modules)

    def forward(self, inputs):
        return self.model(inputs)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, padding_mode='reflect', dropout_prob=0.2):
        super(ConvBlock1D, self).__init__()

        # 1D convolutional layer
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=True)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        conv = self.conv(inputs)
        relu = self.relu(conv)
        dropout = self.dropout(relu)

        return dropout
    
class HarishConvEncoder(nn.Module):
    
    def __init__(self, num_of_features):
        super().__init__()
        kernel_size =3
        padding = 1
        self.encoder = nn.Sequential(
            ConvBlock1D(in_channels=num_of_features,
                        out_channels=32,
                        kernel_size=kernel_size,
                        padding=padding),
            ConvBlock1D(in_channels=32,
                        out_channels=64,
                        kernel_size=kernel_size,
                        padding=padding),
            ConvBlock1D(in_channels=64,
                        out_channels=128,
                        kernel_size=kernel_size,
                        padding=padding)
        )
    
    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        return self.encoder(inputs)
    
    

class DeepConvLstmEncoder(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1))

        self.rnn1 = nn.LSTM(
            input_size=64 * num_of_features,
            hidden_size=embedding_dim,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=0.4,
        )

        def _weights_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):

        inputs = torch.unsqueeze(inputs, dim=1)
        conv1_out = F.relu(self.conv1(inputs))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))

        # print("conv4", conv_4.size())
        reshaped = conv4_out.permute(0, 2, 1, 3).contiguous()
        reshaped = reshaped.view(reshaped.shape[0], reshaped.shape[1], -1)

        rnn1_out, _ = self.rnn1(reshaped)

        return rnn1_out

    def predict_features(self, inputs):
        pass


class CPCEncoder(nn.Module):
    def __init__(
        self, num_of_features: int, embedding_dim: int, num_steps_prediction: int
    ) -> None:

        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        padding_mode = "reflect"

        self.conv1 = nn.Conv1d(
            in_channels=num_of_features,
            out_channels=32,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )

        # RNN to obtain context vector
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=2 * embedding_dim,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=0.2,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.num_steps_prediction = num_steps_prediction

        # def _weights_init(m):
        #     if isinstance(m, (nn.Conv1d, nn.Linear,nn.Conv2d , nn.LSTM, nn.GRU)):
        #         nn.init.xavier_normal_(m.weight)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # self.apply(_weights_init)

    def forward(self, inputs):
        inputs_ = inputs.permute(0, 2, 1)
        inputs_ = self.dropout(self.relu(self.conv1(inputs_)))
        inputs_ = self.dropout(self.relu(self.conv2(inputs_)))
        inputs_ = self.dropout(self.relu(self.conv3(inputs_)))
        # print("output from conv", inputs_.shape)

        z = inputs_.permute(0, 2, 1)

        # print("input to rnn",z.shape)
        # Random timestep to start the future prediction from.
        # If the window is 50 timesteps and k=12, we pick a number from 0-37

        start = torch.randint(
            int(inputs.shape[1] - self.num_steps_prediction), size=(1,)
        ).long()

        # Need to pick the encoded data only until the starting timestep
        rnn_input = z[:, : start + 1, :]

        # Passing through the RNN
        r_out, (_) = self.rnn(rnn_input, None)

        return z, r_out, start

    def predict_features(self, inputs):
        inputs_ = inputs.permute(0, 2, 1)
        inputs_ = self.dropout(self.relu(self.conv1(inputs_)))
        inputs_ = self.dropout(self.relu(self.conv2(inputs_)))
        inputs_ = self.dropout(self.relu(self.conv3(inputs_)))
        z = inputs_.permute(0, 2, 1)

        # Passing through the RNN
        r_out, _ = self.rnn(z, None)

        return r_out

class LstmEncoder(nn.Module):
    def __init__(
        self,
        num_of_features: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=num_of_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
        )

        def _weights_init(m):
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU)):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, inputs):
        # input shape (N,L,H)   N=batch size L= seq len H = input size/ num_of_features

        rnn1_out, (hidden_n, cell_state) = self.rnn1(inputs)

        # output shape (N, L, Hout) Hout = hidden size

        return rnn1_out

    def predict_features(self, inputs):
        pass
