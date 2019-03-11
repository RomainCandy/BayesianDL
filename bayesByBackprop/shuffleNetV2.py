from .basedModel import *
from .basedModel import BatchNorm2DF as BayesianBatchNorm2D

"""
https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenetv2.py
"""


class ShuffleBlock(BayesianNN):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x, kl):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W), kl


class SplitBlock(BayesianNN):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x, kl):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :], kl


class BasicBlock(BayesianNN):
    def __init__(self, in_channels, split_ratio=0.5):
        super(BasicBlock, self).__init__()
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv = BayesianSequential(
            BayesianConv2d(in_channels, in_channels, kernel_size=1, bias=False),
            BayesianBatchNorm2D(in_channels),
            BayesianReLU(),
            BayesianConv2d(in_channels, in_channels,
                           kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            BayesianBatchNorm2D(in_channels),
            BayesianReLU(),
            BayesianConv2d(in_channels, in_channels, kernel_size=1, bias=False),
            BayesianBatchNorm2D(in_channels),
            BayesianReLU())
        self.shuffle = ShuffleBlock()

    def forward(self, x, kl):
        x1, x2, kl = self.split(x, kl)
        out, kl = self.conv(x2, kl)
        out = torch.cat([x1, out], 1)
        out, kl = self.shuffle(out, kl)
        return out, kl


class DownBlock(BayesianNN):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # left
        self.left = BayesianSequential(
            BayesianConv2d(in_channels, in_channels,
                           kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            BayesianBatchNorm2D(in_channels),
            BayesianConv2d(in_channels, mid_channels,
                           kernel_size=1, bias=False),
            BayesianBatchNorm2D(mid_channels),
            BayesianReLU()
        )
        # right
        self.right = BayesianSequential(
            BayesianConv2d(in_channels, mid_channels,
                           kernel_size=1, bias=False),
            BayesianBatchNorm2D(mid_channels),
            BayesianReLU(),
            BayesianConv2d(mid_channels, mid_channels,
                           kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False),
            BayesianBatchNorm2D(mid_channels),
            BayesianConv2d(mid_channels, mid_channels,
                           kernel_size=1, bias=False),
            BayesianBatchNorm2D(mid_channels),
            BayesianReLU(),

        )
        self.shuffle = ShuffleBlock()

    def forward(self, x, kl):
        # left
        out1, kl = self.left(x, kl)
        # right
        out2, kl = self.right(x, kl)
        # concat
        out = torch.cat([out1, out2], 1)
        out, kl = self.shuffle(out, kl)
        return out, kl


class ShuffleNetV2(BayesianModel):
    def __init__(self, net_size, samples=5):
        super(ShuffleNetV2, self).__init__(samples)
        out_channels = configs[net_size]['out_channels']
        num_blocks = configs[net_size]['num_blocks']

        self.conv1 = BayesianSequential(
            BayesianConv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            BayesianBatchNorm2D(24),
            BayesianReLU()
        )
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])

        self.conv2 = BayesianSequential(
            BayesianConv2d(out_channels[2], out_channels[3],
                           kernel_size=1, stride=1, padding=0, bias=False),
            BayesianBatchNorm2D(out_channels[3]),
            BayesianReLU()
        )
        self.linear = BayesianLinear(out_channels[3], 10)

        self.name = f"ShuffleNetV2-{net_size}"

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels))
            self.in_channels = out_channels
        return BayesianSequential(*layers)

    def forward_once(self, x):
        out, kl = self.conv1(x, 0)
        out, kl = self.layer1(out, kl)
        out, kl = self.layer2(out, kl)
        out, kl = self.layer3(out, kl)
        out, kl = self.conv2(out, kl)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out, kl = self.linear(out, kl)
        return out, kl

    def forward(self, x):
        batch_size = x.size(0)
        outputs = torch.zeros(self.samples, batch_size, 10)
        kls = torch.zeros(self.samples)
        for i in range(self.samples):
            outputs[i], kls[i] = self.forward_once(x)
        return outputs.mean(0), kls.mean(0)

    def extra_repr(self):
        return "mle={_mle}, samples={_samples}".format(**self.__dict__)


configs = {
    0.5: {
        'out_channels': (48, 96, 192, 1024),
        'num_blocks': (3, 7, 3)
    },

    1: {
        'out_channels': (116, 232, 464, 1024),
        'num_blocks': (3, 7, 3)
    },
    1.5: {
        'out_channels': (176, 352, 704, 1024),
        'num_blocks': (3, 7, 3)
    },
    2: {
        'out_channels': (224, 488, 976, 2048),
        'num_blocks': (3, 7, 3)
    }
}


def test():
    import torch
    model = ShuffleNetV2(net_size=0.5)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    data = torch.rand(1, 3, 32, 32)
    print(model(data))
    print(model(data))
    model.mle = True
    print(model(data))
    print(model(data))
