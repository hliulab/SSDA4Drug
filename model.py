import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F


# Model of Predictor
class Predictor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=32,
                 drop_out=0.3):
        super(Predictor, self).__init__()

        modules = [nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_out))]

        # in_channels = h_dim

        self.predictor = nn.Sequential(*modules)

    def forward(self, input):
        output = self.predictor(input)
        return output


class Predictor_adentropy(nn.Module):
    def __init__(self, num_class=2, inc=32, temp=0.05, *args, **kwargs):
        super(Predictor_adentropy, self).__init__()
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        # x_out = self.fc(x)
        return x_out


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for m in self.model:
            for name, param in m.named_parameters():
                if param.requires_grad and emb_name in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)  # 默认为2范数
                    if norm != 0:
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for m in self.model:
            for name, param in m.named_parameters():
                if param.requires_grad and emb_name in name:
                    assert name in self.backup
                    param.data = self.backup[name]
        self.backup = {}


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):
        super(MLP, self).__init__()
        modules = []
        hidden_dims = deepcopy(h_dims)

        hidden_dims.insert(0, input_dim)

        self.__in_features = latent_dim

        # Build Encoder
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i - 1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out)
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input):
        embedding = self.encoder(input)
        output = self.bottleneck(embedding)
        return output

    def output_num(self):
        return self.__in_features


class AE(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 h_dims=[512],
                 drop_out=0.3):

        super(AE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = deepcopy(h_dims)

        hidden_dims.insert(0, input_dim)

        # Build Encoder
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i - 1]
            o_dim = hidden_dims[i]

            modules.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )
            # in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                              hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(drop_out))
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2],
                      hidden_dims[-1])
            # , nn.Sigmoid()
        )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        embedding = self.bottleneck(result)

        return embedding

    def decode(self, z):
        """
        Maps the given latent codes
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input):
        embedding = self.encode(input)
        output = self.decode(embedding)
        return embedding, output


class DAE(nn.Module):
    def __init__(self, input_dim=128, fc_dim=256,
                 AE_input_dim=17419, AE_h_dims=[512, 256], pretrained_weights=None, drop=0.1,
                 act='relu'):
        # Construct an autoencoder model
        super(DAE, self).__init__()

        if pretrained_weights is not None:
            self.load_state_dict((torch.load(pretrained_weights)))

        self.__in_features = input_dim

        self.ae = AE(input_dim=AE_input_dim, latent_dim=input_dim, h_dims=AE_h_dims, drop_out=drop)

        self.classifier = Predictor(input_dim=input_dim,output_dim=32)

    def dae(self, data):
        z = data
        y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
        z.data[np.array(y, dtype=bool),] = 0
        embedding = self.ae.encode(z)

        return embedding

    def forward(self, data):
        z = data
        y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
        z.data[np.array(y, dtype=bool),] = 0
        data.requires_grad_(True)
        # print(type(exp)) tensor
        embedding = self.ae.encode(z)
        ae_output = self.ae.decode(embedding)
        # feature = self.self_att_cell(embedding=embedding)
        return embedding, ae_output

    def output_num(self):
        return self.__in_features


class Test_Double_Model(nn.Module):
    def __init__(self, predictor, encoder, adentropy_p):
        super(Test_Double_Model, self).__init__()
        self.predictor = predictor
        self.encoder = encoder
        self.adentropy_p = adentropy_p

    def forward(self, test_data):

        feature = self.encoder(test_data)[0]

        output = self.predictor(feature)
        output = self.adentropy_p(output)

        return output


from torch.autograd import Function


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x,lambd)