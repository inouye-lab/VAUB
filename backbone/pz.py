import torch
import torch.nn as nn

class FixedGaussian(nn.Module):

    def __init__(self, input_dim, scale=1.0, device='cpu'):
        super(FixedGaussian, self).__init__()
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(input_dim).to(device), scale_tril=torch.diag(torch.ones(input_dim)*scale).to(device))

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, n_sample):
        return self.dist.sample((n_sample,))


class MoGNN(nn.Module):

    def __init__(self, input_dim, n_components, loc_init=None, scale_init=None, weight_init=None):

        super(MoGNN, self).__init__()

        if loc_init is None:
            self.loc = nn.Parameter((torch.rand(n_components, input_dim)*2-1)*10)
        else:
            self.loc = nn.Parameter(loc_init)

        if scale_init is None:
            self.log_scale = nn.Parameter(torch.zeros(n_components, input_dim))
        else:
            self.log_scale = nn.Parameter(torch.log(scale_init))

        if weight_init is None:
            self.raw_weight = nn.Parameter(torch.ones(n_components))
        else:
            self.raw_weight = nn.Parameter(torch.log(weight_init/(1-weight_init)))

    def log_prob(self, Z):

        self.loc = self.loc.to(Z.device)
        self.log_scale = self.log_scale.to(Z.device)
        self.raw_weight = self.raw_weight.to(Z.device)

        # print(torch.sigmoid(self.raw_weight))
        mix = torch.distributions.Categorical(torch.sigmoid(self.raw_weight))
        comp = torch.distributions.Independent(torch.distributions.Normal(self.loc, torch.exp(self.log_scale)), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        return gmm.log_prob(Z)
