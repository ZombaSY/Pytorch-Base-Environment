import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def weights_init_normal(m):
    classname = m.__class__.__name__

    # different initialization for each network structure
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)

    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    else:
        print('Undefined structure for initialization :', classname)


class Flatten(nn.Module):
    def forward(self, x):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        return out  # (batch_size, *size)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


# main model
class BaseNet(nn.Module):

    def __init__(self, out_size):
        super().__init__()

        model = [
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            Flatten(),

            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),

            nn.Linear(500, out_size),
            nn.LogSoftmax(dim=1)
        ]

        self.model = nn.Sequential(*model)

        for network in self.model:
            weights_init_normal(network)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(AutoEncoder, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Down-sampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # ----------------------------------------------- LATENT SPACE ----------------------------------------------- #

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Up-sampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
