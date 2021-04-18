import torch
import torch.nn as nn

class LayerDO():
    def __init__(self, in_features, out_features, kernel_size, stride, padding, output_padding = 0):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
    

class Critic(nn.Module):
    def __init__(self, layers:LayerDO):
        super(Critic, self).__init__()
        nn_layers = []
        nn_layers.append(self._conv2D(layers[0]))
        nn_layers.append(nn.LeakyReLU(0.2))
        for layer in layers[1:-1]:
            nn_layers.append(self._block(layer))
        nn_layers.append(self._conv2D(layers[-1]))
        self.critic = nn.Sequential(*nn_layers)
    
    def _conv2D(self, layer:LayerDO):
        return nn.Conv2d(layer.in_features, layer.out_features, layer.kernel_size, layer.stride, layer.padding)

    def _block(self, layer:LayerDO):
        return nn.Sequential(
            nn.Conv2d(layer.in_features, layer.out_features, layer.kernel_size, layer.stride, layer.padding, bias=False),
            nn.InstanceNorm2d(layer.out_features, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.critic(x)


class Generator(nn.Module):
    def __init__(self, layers:LayerDO):
        super(Generator, self).__init__()
        nn_layers = []
        for layer in layers[:-1]:
            nn_layers.append(self._block(layer))
        nn_layers.append(self._convTranspose(layers[-1]))
        nn_layers.append(nn.Tanh())
        self.gen = nn.Sequential(*nn_layers)

    def _convTranspose(self, layer:LayerDO):
            return nn.ConvTranspose2d(layer.in_features, layer.out_features, layer.kernel_size, layer.stride, layer.padding, layer.output_padding)

    def _block(self, layer:LayerDO):
        return nn.Sequential(
            nn.ConvTranspose2d(layer.in_features, layer.out_features, layer.kernel_size, layer.stride, layer.padding, layer.output_padding, bias=False),
            nn.BatchNorm2d(layer.out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():  
        
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    # (in_features, out_features, kernel_size, stride, padding):
    N, in_channels, H, W = 8, 1, 80, 80 
    noise_dim = 100
    
    crit_layers = []
    crit_layers.append(LayerDO(1,64,4,2,1))
    crit_layers.append(LayerDO(64,128,4,2,1))
    crit_layers.append(LayerDO(128,256,4,2,1))
    crit_layers.append(LayerDO(256,512,4,2,1))
    crit_layers.append(LayerDO(512,1,4,2,0))

    gen_layers = []
    gen_layers.append(LayerDO(noise_dim,512,4,1,0))
    gen_layers.append(LayerDO(512,256,4,1,1))
    gen_layers.append(LayerDO(256,128,4,2,1))
    gen_layers.append(LayerDO(128,64,4,2,1))
    gen_layers.append(LayerDO(64,32,4,2,1))
    gen_layers.append(LayerDO(32,1,4,2,1))
    
    x = torch.randn((N, in_channels, H, W))
    print(x.shape)
    critic = Critic(crit_layers)
    print(critic(x).shape)
    assert critic(x).shape == (N, 1, 1, 1), "Critic test failed"
    gen = Generator(gen_layers)
    z = torch.randn((N, noise_dim, 1, 1))
    print(gen(z).shape)
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


# test()
