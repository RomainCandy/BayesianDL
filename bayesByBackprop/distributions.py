import torch
from torch.distributions import Normal
import math


class Posterior:

    def sample(self):
        raise NotImplementedError

    def log_prob(self, inp):
        raise NotImplementedError


class NormalPosterior(Posterior):
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(self.rho.exp())
        # return self.rho

    def sample(self):
        eps = self.normal.sample(self.rho.shape)
        return self.mu + self.sigma * eps

    def log_prob(self, inp):
        res = (-math.log(2 * math.pi) / 2 - self.sigma.log() - ((inp - self.mu) ** 2) / (2 * (self.sigma ** 2))).sum()
        return res


class Prior:

    def log_prob(self, inp):
        raise NotImplementedError


class ScaleMixtureGaussian(Prior):
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)

    def log_prob(self, inp):
        log_prob1 = self.normal1.log_prob(inp)
        log_prob2 = self.normal2.log_prob(inp)
        res = torch.log(1e-16 + self.pi * log_prob1.exp() + (1 - self.pi) * log_prob2.exp()).sum()
        return res


class FixedNormal(Prior):
    # takes mu and logvar as float values and assumes they are shared across all weights
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar
        super(FixedNormal, self).__init__()

    def log_prob(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * math.exp(self.logvar))


if __name__ == '__main__':
    pr = ScaleMixtureGaussian(pi=1, sigma1=math.exp(-2), sigma2=math.exp(-6))
    pr2 = FixedNormal(0, -3)
    data = torch.randn(10)
    print(pr.log_prob(data))
    print(pr2.log_prob(data).sum())
    # n = NormalPosterior(torch.zeros(2000, 3),  torch.ones(2000, 3))
    # m = n.sample()
    # print(m.std(0), m.mean(0))
