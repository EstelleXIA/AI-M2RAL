import torch
import torch.nn as nn


class RNNModule(nn.Module):
    def __init__(self, num_feat, hidden_layers=2, hidden_nodes=256):
        super(RNNModule, self).__init__()

        self.RNN_input_size = num_feat
        self.h_RNN_layers = hidden_layers  # RNN hidden layers
        self.h_RNN = hidden_nodes  # RNN hidden nodes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=hidden_layers,
            batch_first=True,
        )

    def forward(self, x_rnn):
        self.LSTM.flatten_parameters()
        rnn_out, (h_n, h_c) = self.LSTM(x_rnn, None)
        return rnn_out[:, -1, :]


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


class CNNOmicsLabRNNConcatV2(nn.Module):
    def __init__(self, rnn_out_dim=256, fc_dim=128, fc_dim_2=16, drop_p=0.25, num_classes=3):
        super(CNNOmicsLabRNNConcatV2, self).__init__()

        self.omic_sizes = [18, 75, 744, 512]
        sig_networks = []
        for input_dim in self.omic_sizes:
            sig_networks.append(nn.Sequential(*[SNN_Block(dim1=input_dim, dim2=128),
                                                SNN_Block(dim1=128, dim2=128, dropout=0.25)]))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.rnn = RNNModule(num_feat=128, hidden_layers=2, hidden_nodes=64)

        self.h_RNN = rnn_out_dim
        self.h_FC_dim = fc_dim
        self.h_FC_dim_2 = fc_dim_2
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim + 14 + 8 + 32, self.h_FC_dim_2)
        self.fc3 = nn.Linear(self.h_FC_dim_2, self.num_classes)
        self.lab_fc = nn.Linear(29, 8)
        self.order_fc_1 = nn.Linear(512, 128)
        self.order_fc_2 = nn.Linear(128, 8)
        self.texture_fc_1 = nn.Linear(512, 128)
        self.texture_fc_2 = nn.Linear(128, 8)
        self.wavelet_fc_1 = nn.Linear(512, 128)
        self.wavelet_fc_2 = nn.Linear(128, 8)
        self.cnn_fc_1 = nn.Linear(512, 128)
        self.cnn_fc_2 = nn.Linear(128, 8)
        self.drop = nn.Dropout(p=self.drop_p)
        self.relu = nn.ReLU()

    def forward(self, order_input, texture_input, wavelet_input, cnn_input, shape_input, lab_input):

        x_omic = [order_input, texture_input, wavelet_input, cnn_input]
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]

        order_out = self.rnn(h_omic[0])
        texture_out = self.rnn(h_omic[1])
        wavelet_out = self.rnn(h_omic[2])
        cnn_out = self.rnn(h_omic[3])

        lab_out = self.lab_fc(lab_input)
        lab_out = self.relu(lab_out)
        lab_out = self.drop(lab_out)

        order_concat_out = self.order_fc_1(torch.concat([h_omic[1][:, 0], h_omic[1][:, 1], h_omic[1][:, 2], h_omic[1][:, 3]], dim=1))
        order_concat_out = self.relu(order_concat_out)
        order_concat_out = self.drop(order_concat_out)
        order_concat_out = self.order_fc_2(order_concat_out)

        texture_concat_out = self.texture_fc_1(torch.concat([h_omic[0][:, 0], h_omic[0][:, 1], h_omic[0][:, 2], h_omic[0][:, 3]], dim=1))
        texture_concat_out = self.relu(texture_concat_out)
        texture_concat_out = self.drop(texture_concat_out)
        texture_concat_out = self.texture_fc_2(texture_concat_out)

        wavelet_concat_out = self.wavelet_fc_1(torch.concat([h_omic[2][:, 0], h_omic[2][:, 1], h_omic[2][:, 2], h_omic[2][:, 3]], dim=1))
        wavelet_concat_out = self.relu(wavelet_concat_out)
        wavelet_concat_out = self.drop(wavelet_concat_out)
        wavelet_concat_out = self.wavelet_fc_2(wavelet_concat_out)

        cnn_concat_out = self.cnn_fc_1(torch.concat([h_omic[3][:, 0], h_omic[3][:, 1], h_omic[3][:, 2], h_omic[3][:, 3]], dim=1))
        cnn_concat_out = self.relu(cnn_concat_out)
        cnn_concat_out = self.drop(cnn_concat_out)
        cnn_concat_out = self.cnn_fc_2(cnn_concat_out)

        x = self.fc1(torch.concat([order_out, texture_out, wavelet_out, cnn_out], dim=1))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(torch.concat([x, shape_input, lab_out, order_concat_out, texture_concat_out,
                                   wavelet_concat_out, cnn_concat_out], dim=1))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x


class CNNOmicsRNNConcat(nn.Module):
    def __init__(self, rnn_out_dim=256, fc_dim=128, fc_dim_2=16, drop_p=0.25, num_classes=3):
        super(CNNOmicsRNNConcat, self).__init__()

        self.omic_sizes = [18, 75, 744, 512]
        sig_networks = []
        for input_dim in self.omic_sizes:
            sig_networks.append(nn.Sequential(*[SNN_Block(dim1=input_dim, dim2=128),
                                                SNN_Block(dim1=128, dim2=128, dropout=0.25)]))
        self.sig_networks = nn.ModuleList(sig_networks)

        self.rnn = RNNModule(num_feat=128, hidden_layers=2, hidden_nodes=64)

        self.h_RNN = rnn_out_dim
        self.h_FC_dim = fc_dim
        self.h_FC_dim_2 = fc_dim_2
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim + 14 + 32, self.h_FC_dim_2)
        self.fc3 = nn.Linear(self.h_FC_dim_2, self.num_classes)
        self.order_fc_1 = nn.Linear(512, 128)
        self.order_fc_2 = nn.Linear(128, 8)
        self.texture_fc_1 = nn.Linear(512, 128)
        self.texture_fc_2 = nn.Linear(128, 8)
        self.wavelet_fc_1 = nn.Linear(512, 128)
        self.wavelet_fc_2 = nn.Linear(128, 8)
        self.cnn_fc_1 = nn.Linear(512, 128)
        self.cnn_fc_2 = nn.Linear(128, 8)
        self.drop = nn.Dropout(p=self.drop_p)
        self.relu = nn.ReLU()

    def forward(self, order_input, texture_input, wavelet_input, cnn_input, shape_input, lab_input):

        x_omic = [order_input, texture_input, wavelet_input, cnn_input]
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]

        order_out = self.rnn(h_omic[0])
        texture_out = self.rnn(h_omic[1])
        wavelet_out = self.rnn(h_omic[2])
        cnn_out = self.rnn(h_omic[3])

        order_concat_out = self.order_fc_1(torch.concat([h_omic[1][:, 0], h_omic[1][:, 1], h_omic[1][:, 2], h_omic[1][:, 3]], dim=1))
        order_concat_out = self.relu(order_concat_out)
        order_concat_out = self.drop(order_concat_out)
        order_concat_out = self.order_fc_2(order_concat_out)

        texture_concat_out = self.texture_fc_1(torch.concat([h_omic[0][:, 0], h_omic[0][:, 1], h_omic[0][:, 2], h_omic[0][:, 3]], dim=1))
        texture_concat_out = self.relu(texture_concat_out)
        texture_concat_out = self.drop(texture_concat_out)
        texture_concat_out = self.texture_fc_2(texture_concat_out)

        wavelet_concat_out = self.wavelet_fc_1(torch.concat([h_omic[2][:, 0], h_omic[2][:, 1], h_omic[2][:, 2], h_omic[2][:, 3]], dim=1))
        wavelet_concat_out = self.relu(wavelet_concat_out)
        wavelet_concat_out = self.drop(wavelet_concat_out)
        wavelet_concat_out = self.wavelet_fc_2(wavelet_concat_out)

        cnn_concat_out = self.cnn_fc_1(torch.concat([h_omic[3][:, 0], h_omic[3][:, 1], h_omic[3][:, 2], h_omic[3][:, 3]], dim=1))
        cnn_concat_out = self.relu(cnn_concat_out)
        cnn_concat_out = self.drop(cnn_concat_out)
        cnn_concat_out = self.cnn_fc_2(cnn_concat_out)

        x = self.fc1(torch.concat([order_out, texture_out, wavelet_out, cnn_out], dim=1))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(torch.concat([x, shape_input, order_concat_out, texture_concat_out,
                                   wavelet_concat_out, cnn_concat_out], dim=1))
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x


class CNNConcat(nn.Module):
    def __init__(self, f_dim=512, dropout_prob=0.5, num_classes=3):
        super(CNNConcat, self).__init__()

        self.fc_all = torch.nn.Sequential(
            nn.Linear(f_dim * 3, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(f_dim, f_dim),
            nn.Linear(f_dim, num_classes)
        )

    def forward(self, order_input, texture_input, wavelet_input, cnn_input, shape_input, lab_input):
        concatenated_features = torch.concat([cnn_input[:, 0], cnn_input[:, 1], cnn_input[:, 2]], dim=1)
        x = self.fc_all(concatenated_features)

        return x
