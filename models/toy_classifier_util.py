import torch
import torch.nn as nn
from torch.distributions import multivariate_normal as dist_mn
from sklearn import mixture
import matplotlib as mpl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np


class ToyDataset:

    def __init__(self, domain, label, n_sample, data_config, device='cpu'):

        self.domain = torch.tensor(domain).to(device)
        self.label = torch.tensor(label).to(device)
        self.data = self.getdata(data_config, n_sample, device=device)
        # self.dist = None

    # @staticmethod
    def getdata(self, config, n_sample, device='cpu'):

        if config['type'] == 'Gaussian':
            self.dist = dist_mn.MultivariateNormal(config['mean'].to(device), config['cov'].to(device))
            return self.dist.sample((n_sample,))

        else:
            print("Invalid config type")
            return -1


class LinearLayer(nn.Module):

    def __init__(self, in_feature, out_feature, bias=True):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(in_feature, out_feature, bias=bias)
        nn.init.eye_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0)
        self.det = None

    def forward(self, x):
        self.det = torch.slogdet(self.layer.weight)[1]
        y = self.layer(x)
        return y


class IdentityLayer(nn.Module):

    def __init__(self):
        super(IdentityLayer, self).__init__()
        self.layer = nn.Identity()
        self.det = None

    def forward(self, x):
        self.det = 0
        y = self.layer(x)
        return y


class ReLULayer(nn.Module):

    def __init__(self, negative_slope=0.5):
        super(ReLULayer, self).__init__()
        self.layer = nn.LeakyReLU(negative_slope=negative_slope)
        self.negative_slope = negative_slope
        self.det = None

    def forward(self, x):
        self.det = torch.sum(x < 0) * np.log(self.negative_slope) / x.size(0)
        y = self.layer(x)
        return y


class EasyFCLayer(nn.Module):

    def __init__(self, n_features=2, n_layer=2):
        super(EasyFCLayer, self).__init__()
        self.layer = nn.ModuleList([
            LinearLayer(n_features, n_features, bias=True),
            ReLULayer(),
            LinearLayer(n_features, n_features, bias=True),
            ReLULayer(),
            # LinearLayer(n_features, n_features, bias=True),
            # ReLULayer(),
            # LinearLayer(n_features, n_features, bias=True),
            # ReLULayer(),
            # LinearLayer(n_features, n_features, bias=True),
            # ReLULayer(),
            # LinearLayer(n_features, n_features, bias=True),
            # ReLULayer(),
            LinearLayer(n_features, n_features, bias=True)]
        )
        self.det = None

    def forward(self, x):
        self.det = 0
        for T in self.layer:
            x = T(x)
            self.det += T.det
        return x


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.det = None

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
            #             log_det_J += s.sum(dim=1)
            self.det = log_det_J.mean()
        #         return z, log_det_J
        return z

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


class QZmodel:

    def __init__(self, args):

        self.args = args

        if self.args['name'] == 'MoG':
            self.QZ = mixture.GaussianMixture(n_components=args['n_components'],
                                              covariance_type=args['covariance_type'],
                                              warm_start=args['warm_start'],
                                              reg_covar=args['reg_covar'])
        elif self.args['name'] == 'Gaussian':
            self.QZ = None  # getting defined in update_QZ function

    def _toTensor(self, X):

        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).type(torch.FloatTensor)

    def gmm_draw_ellipse(self, data, ax, color):

        QZ_temp = mixture.GaussianMixture(n_components=self.args['n_components'],
                                          covariance_type=self.args['covariance_type']).fit(data)

        for n in range(QZ_temp.weights_.shape[0]):
            if self.args['covariance_type'] == 'diag':
                cov = np.diag(QZ_temp.covariances_[n])
            else:
                cov = QZ_temp.covariances_[n]
            #             print(cov.shape)
            covariances = cov[:2, :2]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.patches.Ellipse(QZ_temp.means_[n, :2], v[0], v[1],
                                      180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    #             ax.set_aspect('equal', 'datalim')

    def update_QZ(self, Z):

        if self.args['name'] == 'MoG':
            self.QZ.fit(Z)

        elif self.args['name'] == 'Gaussian':
            loc = torch.mean(Z, dim=0)
            Z_centered = Z - loc
            cov = torch.mm(Z_centered.t(), Z_centered) / Z.shape[0]
            self.QZ = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

    def log_prob(self, Z):

        if self.args['name'] == 'MoG':
            score = torch.zeros(Z.shape[0])
            for idx in range(self.args['n_components']):

                #                 print(self.QZ.covariances_[idx])
                loc = self._toTensor(self.QZ.means_[idx])
                if self.args['covariance_type'] == 'diag':
                    cov = self._toTensor(np.diag(self.QZ.covariances_[idx]))
                else:
                    cov = self._toTensor(self.QZ.covariances_[idx])

                QZ_temp = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)

                score += self.QZ.weights_[idx] * QZ_temp.log_prob(Z).exp_()

            #             print (torch.mean(torch.log(score)).detach(), self.QZ.score(Z.detach()))
            return torch.mean(torch.log(score))
        elif self.args['name'] == 'Gaussian':
            return torch.mean(self.QZ.log_prob(Z))


class LinearClassifier(nn.Module):

    def __init__(self, hid_dim=16):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calculate_loss(self, X, y, criterion):
        outputs = self.forward(X)
        return criterion(outputs, y)


class AUBmodel(nn.Module):

    def __init__(self, T_list, QZ, device='cpu'):
        super(AUBmodel, self).__init__()

        self.device = device
        self.QZ = QZ
        #         self._set_input(X_list, T_list)
        self.T_list = T_list

    #     def _set_input(self, X_list, T_list):
    # move to gpu if necessary
    #         self.X_list = X_list
    #         self.T_list = T_list

    def _forward_z(self):

        self.Z_list = [
            T(X)
            for X, T in zip(self.X_list, self.T_list)
        ]

    def _calculate_QZ(self):

        # Fit QZ density (detach since we are not optimizing this directly)
        #  (if deep density model, then don't detach)
        Z_mix = torch.cat([Z.detach() for Z in self.Z_list], dim=0)
        self.QZ.update_QZ(Z_mix)

    def calculate_tc_loss(self, X_list, weight=None, option='L2'):
        loss = 0
        for X, T in zip(X_list, self.T_list):
            diff = torch.abs(X - T(X))
            if weight is None:  # default is uniform weight
                weight = 1 / len(X_list)
            loss += weight * torch.mean(torch.sum(diff * diff, dim=1))
        if option == 'L22':
            return loss
        elif option == 'L2':
            return torch.sqrt(loss)

    def calculate_loss(self, X_list, y_list=None, updateQ=True):

        self.Z_list = [T(X) for X, T in zip(X_list, self.T_list)]
        self.logdet = []

        if updateQ is True:
            self._calculate_QZ()

        # Now compute actual loss function
        loss = 0
        if y_list is None:
            for Z, T in zip(self.Z_list, self.T_list):
                # Log det term
                if T.det is not None:
                    self.logdet.append(T.det.detach())
                    loss -= T.det  # Extract value
                else:
                    raise RuntimeError('determinant is not calculated yet')
                # log Qz term
                loss -= self.QZ.log_prob(Z)
        else:
            for Z, T, y in zip(self.Z_list, self.T_list, y_list):
                # Log det term
                if T.det is not None:
                    self.logdet.append(T.det.detach())
                    loss -= T.det  # Extract value
                else:
                    raise RuntimeError('determinant is not calculated yet')
                # log Qz term
                loss -= self.QZ.log_prob(Z, y)
        return loss

    def forward(self, X_list):
        self.Z_list = [T(X) for X, T in zip(X_list, self.T_list)]
        return self.Z_list

#     def train_iter(self, lam, optimizer, epoch, updateQstep=1):

#         for T in T_list:
#             T.train()

#         if epoch % updateQstep == 0:
#             updateQ = True
#         else:
#             updateQ = False

#     #     print(f'epoch {epoch}, update QZ is {updateQ}')
#         optimizer.zero_grad()
#         loss_pfl = self._pushforward_loss(updateQ = updateQ)
#         loss = lam * loss_pfl
#         loss_tpc = self._transportation_cost()
#         loss += loss_tpc
#         loss.backward()
#         optimizer.step()

#         return loss_pfl, loss_tpc, loss


def compute_entropy(log_prob):
    return -1*torch.sum(log_prob.exp()*log_prob)

def compute_kl(log_p, log_q):
    return -1*torch.sum(log_p.exp()*np.log(log_q.exp()/log_p.exp()))

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class SinkhornDistance(nn.Module):
    # from https://github.com/dfdazac/wassdistance/blob/master/layers.py
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


def part_wd(X, Xp, partition=1000, norm_var=False, eps=1e-4, max_iter=100):
    sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)
    n_samples = X.shape[0]
    idx = 0
    avg_wd = 0
    while idx < n_samples - partition:
        avg_wd += partition * sinkhorn(X[idx:idx + partition], Xp[idx:idx + partition])
        idx += partition

    partition = n_samples - idx
    avg_wd += partition * sinkhorn(X[idx:idx + partition], Xp[idx:idx + partition])

    if norm_var:
        avg_wd = avg_wd / torch.var(X)

    return avg_wd / n_samples


def show_train(Loss, T_list, x_data, options='all'):

    if options == 'all' or options == 'loss_only':

        # Visualize Loss Graph
        fig, ax = plt.subplots(4,1,figsize=(16,24))
        ax[0].plot(Loss["Loss_all"])
        ax[1].plot(Loss["Loss_aub"])
        ax[2].plot(Loss["Loss_cls"])
        ax[3].plot(Loss["Loss_tc"])
        ax[0].set_title('Overall loss')
        ax[1].set_title('AUB loss')
        ax[2].set_title('Classifier loss')
        ax[3].set_title('Transportation cost')
        plt.show()

    if options == 'all' or options == 'dataset_only':

        # Visualize transformed dataset
        fig, ax = plt.subplots(figsize=(8,8))
        Z_list = [T_list[0](x_data["d0c0"].data).detach(), T_list[1](x_data["d1c0"].data).detach(), T_list[0](x_data["d0c1"].data).detach(), T_list[1](x_data["d1c1"].data).detach()]
        # len(Z_list)
        ax.scatter(*Z_list[0].T.cpu(), label='z_d0_c0', c='b', marker='o')
        ax.scatter(*Z_list[1].T.cpu(), label='z_d1_c0', c='r', marker='o')
        ax.scatter(*Z_list[2].T.cpu(), label='z_d0_c1', c='b', marker='+')
        ax.scatter(*Z_list[3].T.cpu(), label='z_d1_c1', c='r', marker='+')
        plt.legend()
        plt.show()


def train_caub(x_data, aub_model, cls_model, n_epoch, n_sample, epsilon, lambda_cls, lambda_aub, lambda_tc, loss_option, lr_cls, lr_aub, device):

    aub_model.to(device)
    cls_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_cls = optim.RMSprop(cls_model.parameters(), lr=lr_cls)
    optimizer_aub = optim.RMSprop(aub_model.parameters(), lr=lr_aub)

    aub_model.train()
    cls_model.train()

    Loss = {"Loss_all": [], "Loss_aub": [], "Loss_cls":[], "Loss_tc":[]}
    # Loss_all, Loss_aub, Loss_cls, Loss_tc = [], [], [], []

    for epoch in range(n_epoch):
        optimizer_cls.zero_grad()
        optimizer_aub.zero_grad()

        X_list = [torch.cat([x_data["d0c0"].data, x_data["d0c1"].data]), torch.cat([x_data["d1c0"].data, x_data["d1c1"].data])]
        # X_list = [X.to(device) for X in X_list]
        idx = torch.randperm(n_sample//2)
        X_list = [X[idx] for X in X_list]
        loss_aub = aub_model.calculate_loss(X_list)

        loss_tc = aub_model.calculate_tc_loss(X_list, option=loss_option)

        X_list = [torch.cat([x_data["d0c0"].data, x_data["d0c1"].data]), torch.cat([x_data["d1c0"].data, x_data["d1c1"].data])]
        X_list = [X.to(device) for X in X_list]
        Z_list = aub_model(X_list)
        Z = torch.cat(Z_list)
        # y = torch.cat([torch.tensor([label]*(n_sample//4)) for label in [0,1,0,1]])
        # y = torch.cat([[label]*(n_sample//4) for label in [0,1,0,1]])
        y = torch.tensor([[0], [1], [0], [1]], device=device).expand(-1, n_sample//4).reshape(-1)
        idx = torch.randperm(n_sample)
        Z, y = Z[idx], y[idx]
        # Z, y = Z[idx], y[idx].to(device)
        loss_cls = cls_model.calculate_loss(Z, y, criterion)

    #     loss = loss_cls + lambda_aub*loss_aub
        loss_all = lambda_cls*loss_cls + lambda_aub*max(loss_aub-epsilon,0) + lambda_tc*loss_tc

        loss_all.backward()
        optimizer_cls.step()
        optimizer_aub.step()

        Loss["Loss_all"].append(loss_all.item())
        Loss["Loss_aub"].append(loss_aub.item())
        Loss["Loss_cls"].append(loss_cls.item())
        Loss["Loss_tc"].append(loss_tc.item())

        # if epoch%(n_epoch//100) == 0:
        #     wd = part_wd(Z_list[0], Z_list[1])

        if epoch%(n_epoch//10) == 0:
            print(f"Epoch {epoch}/{n_epoch}  Loss:{loss_all.item():02f}  Loss_aub:{loss_aub.item():02f}  Loss_cls:{loss_cls.item():02f}  Loss_tc:{loss_tc.item():02f}")

    return Loss


def get_sigmas(sigma_begin, sigma_end, sigma_steps, sigma_dist='geometric'):
    if sigma_dist == 'geometric':
        sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), sigma_steps))).float() # same thing as np.logspace
    elif sigma_dist == 'uniform':
        sigmas = torch.tensor(np.linspace(sigma_begin, sigma_end, sigma_steps)).float()

    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas

def train_caub_with_noise(x_data, sigmas, aub_model, cls_model, n_epoch, n_sample, epsilon, lambda_cls, lambda_aub, lambda_tc, loss_option, lr_cls, lr_aub, device, tc_lambdas=None):

    aub_model.to(device)
    cls_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_cls = optim.RMSprop(cls_model.parameters(), lr=lr_cls)
    optimizer_aub = optim.RMSprop(aub_model.parameters(), lr=lr_aub)

    aub_model.train()
    cls_model.train()

    Loss = {"Loss_all": [], "Loss_aub": [], "Loss_cls":[], "Loss_tc":[]}
    x_data_noisy = {}
    # Loss_all, Loss_aub, Loss_cls, Loss_tc = [], [], [], []

    for epoch in range(n_epoch):
        optimizer_cls.zero_grad()
        optimizer_aub.zero_grad()

        for k,v in x_data.items():
            x_data_noisy[k] = v.data + sigmas[epoch]*torch.randn_like(v.data)

        X_list = [torch.cat([x_data_noisy["d0c0"], x_data_noisy["d0c1"]]), torch.cat([x_data_noisy["d1c0"], x_data_noisy["d1c1"]])]
        # X_list = [X.to(device) for X in X_list]
        idx = torch.randperm(n_sample//2)
        X_list = [X[idx] for X in X_list]
        loss_aub = aub_model.calculate_loss(X_list)

        loss_tc = aub_model.calculate_tc_loss(X_list, option=loss_option)

        X_list = [torch.cat([x_data["d0c0"].data, x_data["d0c1"].data]), torch.cat([x_data["d1c0"].data, x_data["d1c1"].data])]
        X_list = [X.to(device) for X in X_list]
        Z_list = aub_model(X_list)
        Z = torch.cat(Z_list)
        # y = torch.cat([torch.tensor([label]*(n_sample//4)) for label in [0,1,0,1]])
        # y = torch.cat([[label]*(n_sample//4) for label in [0,1,0,1]])
        y = torch.tensor([[0], [1], [0], [1]], device=device).expand(-1, n_sample//4).reshape(-1)
        idx = torch.randperm(n_sample)
        Z, y = Z[idx], y[idx]
        # Z, y = Z[idx], y[idx].to(device)
        loss_cls = cls_model.calculate_loss(Z, y, criterion)

    #     loss = loss_cls + lambda_aub*loss_aub
        if tc_lambdas is None:
            loss_all = lambda_cls*loss_cls + lambda_aub*max(loss_aub-epsilon,0) + lambda_aub*loss_tc
        else:
            ratio = (lambda_aub*max(loss_aub-epsilon,0)).item()/loss_tc.item()
            loss_all = lambda_cls*loss_cls + lambda_aub*max(loss_aub-epsilon,0) + ratio*tc_lambdas[epoch]*loss_tc

        loss_all.backward()
        optimizer_cls.step()
        optimizer_aub.step()

        Loss["Loss_all"].append(loss_all.item())
        Loss["Loss_aub"].append(loss_aub.item())
        Loss["Loss_cls"].append(loss_cls.item())
        Loss["Loss_tc"].append(loss_tc.item())

        # if epoch%(n_epoch//100) == 0:
        #     wd = part_wd(Z_list[0], Z_list[1])

        if epoch%(n_epoch//10) == 0:
            # print((lambda_aub*max(loss_aub-epsilon,0)).item(), (ratio*tc_lambdas[epoch]*loss_tc).item())
            print(f"Epoch {epoch}/{n_epoch}  Loss:{loss_all.item():02f}  Loss_aub:{loss_aub.item():02f}  Loss_cls:{loss_cls.item():02f}  Loss_tc:{loss_tc.item():02f}")

    return Loss

def get_tc_lambdas(tc_begin, tc_end, tc_steps, tc_dist='geometric'):
    if tc_dist == 'geometric':
        tc_lambdas = torch.tensor(np.exp(np.linspace(np.log(tc_begin), np.log(tc_end), tc_steps))).float()
    elif tc_dist == 'uniform':
        tc_lambdas = torch.tensor(np.linspace(tc_begin, tc_end, tc_steps)).float()

    else:
        raise NotImplementedError('tc distribution not supported')
    return tc_lambdas