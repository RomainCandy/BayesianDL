import torch
from torch.distributions import Normal
import math


class Posterior:

    def sample(self):
        raise NotImplementedError

    def log_prob(self, inp):
        raise NotImplementedError


class NormalPosterior(Posterior):
    def __init__(self, mu, rho, device):
        self.mu = mu.to(device)
        self.rho = rho.to(device)
        self.normal = Normal(torch.zeros(1).to(device), torch.ones(1).to(device))

    @property
    def sigma(self):
        return torch.log1p(self.rho.exp())

    def sample(self):
        eps = self.normal.sample(self.rho.shape).squeeze(-1)
        return self.mu + self.sigma * eps

    def log_prob(self, inp):
        res = (-math.log(2 * math.pi) / 2 - self.sigma.log() - ((inp - self.mu) ** 2) / (2 * (self.sigma ** 2))).sum()
        return res


class Prior:

    def log_prob(self, inp):
        raise NotImplementedError


class ScaleMixtureGaussian(Prior):
    def __init__(self, pi, sigma1, sigma2, device):
        self.pi = pi
        if isinstance(sigma1, float):
            self.sigma1 = torch.Tensor([sigma1]).to(device)
        else:
            self.sigma1 = sigma1.to(device)
        if isinstance(sigma2, float):
            self.sigma1 = torch.Tensor([sigma2]).to(device)
        else:
            self.sigma2 = sigma2.to(device)
        # self.sigma2 = sigma2.to(device)
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)

    def log_prob(self, inp):
        log_prob1 = self.normal1.log_prob(inp)
        log_prob2 = self.normal2.log_prob(inp)
        res = torch.log(1e-16 + self.pi * log_prob1.exp() + (1 - self.pi) * log_prob2.exp()).sum()
        return res


class FixedNormal(Prior):
    # takes mu and logvar as float values and assumes they are shared across all weights
    def __init__(self, mu, logvar, device):
        self.mu = mu.to(device)
        self.logvar = logvar.to(device)
        super(FixedNormal, self).__init__()

    def log_prob(self, x):
        c = - float(0.5 * math.log(2 * math.pi))
        return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * math.exp(self.logvar))


if __name__ == '__main__':
    pr = ScaleMixtureGaussian(pi=1, sigma1=math.exp(-2), sigma2=math.exp(-6), device="cpu")
    pr2 = FixedNormal(0, -3, "cpu")
    data = torch.randn(10)
    print(pr.log_prob(data))
    print(pr2.log_prob(data).sum())
    # n = NormalPosterior(torch.zeros(2000, 3),  torch.ones(2000, 3))
    # m = n.sample()
    # print(m.std(0), m.mean(0))
