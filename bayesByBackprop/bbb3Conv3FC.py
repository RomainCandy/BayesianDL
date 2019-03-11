from .basedModel import *


class BBB3Conv3FC(BayesianNN):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, channels_in, classes):
        super(BBB3Conv3FC, self).__init__()
        self.model = BayesianSequential(
            BayesianConv2d(channels_in, 32, 5, stride=1, padding=2),
            BayesianReLU(),
            BayesianMaxPool2d(kernel_size=3, stride=2),
            BayesianConv2d(32, 64, 5, stride=1, padding=2),
            BayesianReLU(),
            BayesianMaxPool2d(kernel_size=3, stride=2),
            BayesianConv2d(64, 128, 5, stride=1, padding=1),
            BayesianReLU(),
            BayesianMaxPool2d(kernel_size=3, stride=2),
            FlattenLayer(2 * 2 * 128),
            BayesianLinear(2 * 2 * 128, 100),
            BayesianReLU(),
            BayesianLinear(100, 100),
            BayesianReLU(),
            BayesianLinear(100, classes)
        )

    def forward(self, x):
        return self.model(x, 0)
