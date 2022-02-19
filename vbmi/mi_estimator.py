import torch
import math
from torch import nn
from torch import cuda
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


#Snippet of code written to take benefits of GPU computation
train_on_GPU = cuda.is_available()
print(f'training on GPU: {train_on_GPU}')
# device = 'cuda' if train_on_GPU else 'cpu'
# assert train_on_GPU, "please run this code on GPU (Google Collab is free and sufficient for instance)"


# Separate Critic Implementation
class MINE(nn.Module):
    def __init__(self, input_dim, zdim):
        super(MINE, self).__init__()

        self.input_dim = input_dim
        self.zdim = zdim
        self.moving_average = None

        self.MLP_g = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
        self.MLP_h = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, x, z):
        x = x.view(-1, self.input_dim)
        z = z.view(-1, self.zdim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(z)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1))

        return scores  # Each element i,j is a scalar in R. f(xi,proj_j)


class baseline_MLP(nn.Module):
    def __init__(self, input_dim):
        super(baseline_MLP, self).__init__()

        self.input_dim = input_dim

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        res = self.MLP(x)

        # Output a scalar which is the log-baseline : log a(y)  for interpolated bound
        return res


def reduce_logmeanexp_nodiag(x, device, axis=None):
    batch_size = x.size()[0]
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)),dim=[0,1])
    num_elem = batch_size * (batch_size - 1.)
    return logsumexp - torch.log(torch.tensor(num_elem).to(device))


def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:,None]
    joint_term= torch.mean(torch.diag(scores))
    marg_term=torch.exp(reduce_logmeanexp_nodiag(scores))
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
    return tuba_lower_bound(scores - 1.)


#Compute the Noise Constrastive Estimation (NCE) loss
def infonce_lower_bound(scores):
    '''Bound from Van Den Oord and al. (2018)'''
    nll = torch.mean( torch.diag(scores) - torch.logsumexp(scores,dim=1))
    k =scores.size()[0]
    mi = np.log(k) + nll
    return mi


def log_interpolate(log_a, log_b, alpha_logit, device):
    '''Numerically stable implmentation of log(alpha * a + (1-alpha) *b)
    Compute the log baseline for the interpolated bound
    baseline is a(y)'''
    log_alpha = -F.softplus(torch.tensor(-alpha_logit).to(device))
    log_1_minus_alpha = -F.softplus(torch.tensor(alpha_logit).to(device))
    y = torch.logsumexp(torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b)), dim=0)
    return y


def compute_log_loomean(scores, device):
    '''Compute the log leave one out mean of the exponentiated scores'''
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)

    lse_minus_max = torch.logsumexp(scores - max_scores, dim=1, keepdim=True)
    d = lse_minus_max + (max_scores - scores)

    d_not_ok = torch.eq(d, 0.)
    d_ok = ~d_not_ok
    safe_d = torch.where(d_ok, d, torch.ones_like(d).to(device))  # Replace zeros by 1 in d

    loo_lse = scores + (safe_d + torch.log(-torch.expm1(-safe_d)))  # Stable implementation of softplus_inverse
    loo_lme = loo_lse - np.log(scores.size()[1] - 1.)
    return loo_lme


# Compute interporlate lower bound of MI
def interpolated_lower_bound(scores, baseline, alpha_logit):
    '''
    New lower bound on mutual information proposed by Ben Poole and al.
    in "On Variational Bounds of Mutual Information"
    It allows to explictily control the biais-variance trade-off.
    For MI estimation -> This bound with a small alpha is much more stable but
    still small biais than NWJ / Mine-f bound !
    Return a scalar, the lower bound on MI
    '''
    batch_size = scores.size()[0]
    nce_baseline = compute_log_loomean(scores)

    interpolated_baseline = log_interpolate(nce_baseline,
                                            baseline[:, None].repeat(1, batch_size),
                                            alpha_logit)  # Interpolate NCE baseline with a learnt baseline

    # Marginal distribution term
    critic_marg = scores - torch.diag(interpolated_baseline)[:, None]  # None is equivalent to newaxis
    marg_term = torch.exp(reduce_logmeanexp_nodiag(critic_marg))

    # Joint distribution term
    critic_joint = torch.diag(scores)[:, None] - interpolated_baseline
    joint_term = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / (batch_size * (batch_size - 1.))
    return 1 + joint_term - marg_term


def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None):
      """Estimate variational lower bounds on mutual information.

      Args:
        estimator: string specifying estimator, one of:
          'nwj', 'infonce', 'tuba', 'js', 'interpolated'
        x: [batch_size, dim_x] Tensor
        y: [batch_size, dim_y] Tensor
        critic_fn: callable that takes x and y as input and outputs critic scores
          output shape is a [batch_size, batch_size] matrix
        baseline_fn (optional): callable that takes y as input
          outputs a [batch_size]  or [batch_size, 1] vector
        alpha_logit (optional): logit(alpha) for interpolated bound

      Returns:
        scalar estimate of mutual information
      """
      scores = critic_fn(x, y)
      if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
      if estimator == 'infonce':
        mi = infonce_lower_bound(scores)
      elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
      elif estimator == 'tuba':
        mi = tuba_lower_bound(scores, log_baseline)
      elif estimator == 'interpolated':
        assert alpha_logit is not None, "Must specify alpha_logit for interpolated bound."
        mi = interpolated_lower_bound(scores, log_baseline, alpha_logit)
      return mi


def mlp(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu'):
    if activation == 'relu':
        activation_f = nn.ReLU()
    layers = [nn.Linear(input_dim, hidden_dim), activation_f]
    for _ in range(n_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation_f]
    layers += [nn.Linear(hidden_dim, output_dim)]
    # import pdb; pdb.set_trace()
    return nn.Sequential(*layers)


class SeparableCritic(nn.Module):
    def __init__(self, x_dim, y_dim, embed_dim, n_layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(x_dim, embed_dim, n_layers, activation)
        self._h = mlp(y_dim, embed_dim, n_layers, activation)

    def forward(self, x, y):
        x = x.view(-1, self.x_dim)
        y = y.view(-1, self.y_dim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(y)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1)) #Each element i,j is a scalar in R. f(xi,proj_j)
        return scores


class ConcatCritic(nn.Module):
    def __init__(self, x_dim, y_dim, embed_dim, n_layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(x_dim+y_dim, embed_dim, 1, n_layers, activation)

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Tile all possible combinations of x and y
        x_tiled = torch.tile(x[None, :],  (batch_size, 1, 1))
        y_tiled = torch.tile(y[:, None],  (1, batch_size, 1))
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.transpose(torch.reshape(scores, [batch_size, batch_size]), 1, 0)


class UnnormalizedBaseline(nn.Module):
    def __init__(self, input_dim, embed_dim=512, n_layers=1, activation='relu', **extra_kwargs):
        super(UnnormalizedBaseline, self).__init__()
        # output is scalar score
        self.input_dim = input_dim
        self._f = mlp(input_dim, embed_dim, 1, n_layers, activation)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        scores = self._f(x)
        return scores


CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic
}


BASELINES= {
    'constant': lambda: None,
    'unnormalized': UnnormalizedBaseline
}


class MIEstimator(object):
    def __init__(self, critic_params, data_params, mi_params, opt_params, device):
        self.mi_params = mi_params
        self.device = device
        self.critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, **critic_params)
        self.critic.to(self.device)
        # import pdb; pdb.set_trace()
        if mi_params.get('baseline', 'constant') == "constant":
            self.baseline = BASELINES[mi_params.get('baseline', 'constant')]()
        else:
            self.baseline = BASELINES[mi_params.get('baseline', 'constant')](input_dim=data_params['dim'])
            self.baseline.to(self.device)
        if self.baseline is not None:
            self.trainable_vars = list(self.critic.parameters()) + list(self.baseline.parameters())
        else:
            self.trainable_vars = list(self.critic.parameters())
        self.optimizer = optim.Adam(self.trainable_vars, lr=opt_params['learning_rate'])

    def fit(self, dataloader, epochs=50):
        """
        :param x: array [num_data, dim_x] Representing a dataset of samples from P_X
        :param y: array [num_data, dim_y] Representing a dataset of samples from P_Y|X=x
        :param epochs:
        :return:
        """
        history_MI = []
        for epoch in tqdm(range(epochs)):
            MI_epoch = 0
            for i_batch, sample_batch in enumerate(dataloader):
                # import pdb; pdb.set_trace()
                x, y = sample_batch

                x = x.float().to(self.device)
                y = y.float().to(self.device)
                mi = estimate_mutual_information(self.mi_params['estimator'], x, y, self.critic, self.baseline,
                                                 self.mi_params.get('alpha_logit'))
                MI_loss = -mi
                self.optimizer.zero_grad()
                MI_loss.backward()
                self.optimizer.step()
                MI_epoch += mi
            MI_epoch /= 50
            history_MI.append(MI_epoch.detach().cpu().numpy())
        print('Finished Training')
        return np.asarray(history_MI)


def train_estimator(critic_params, data_params, mi_params, opt_params, device):
    """Main training loop that estimates time-varying MI."""
    # Ground truth rho is only used by conditional critic
    critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, **critic_params)
    critic.to(device)
    # import pdb; pdb.set_trace()
    if mi_params.get('baseline', 'constant') == "constant":
        baseline = BASELINES[mi_params.get('baseline', 'constant')]()
    else:
        baseline = BASELINES[mi_params.get('baseline', 'constant')](input_dim=data_params['dim'])
        baseline.to(device)

    if baseline is not None:
        trainable_vars = list(critic.parameters()) + list(baseline.parameters())
    else:
        trainable_vars = list(critic.parameters())
    optimizer = optim.Adam(trainable_vars, lr=opt_params['learning_rate'])

    history_MI = []
    for epoch in range(opt_params['n_epochs']):
        MI_epoch = 0
        for i in range(50):  # Mimic a dataset of size 50*Batch_Size
            x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=data_params['rho'], batch_size=data_params['batch_size'])
            x = x.to(device)
            y = y.to(device)
            mi = estimate_mutual_information(mi_params['estimator'], x, y, critic, baseline, mi_params.get('alpha_logit'))
            MI_loss = -mi

            optimizer.zero_grad()
            MI_loss.backward()
            optimizer.step()

            MI_epoch += mi

        MI_epoch /= 50
        history_MI.append(MI_epoch.detach().cpu().numpy())
        # lr_scheduler.step()

        if (epoch+1) % 5 == 0:
            print('==========> Epoch: {} ==========> MI: {:.4f}'.format(epoch+1, MI_epoch))

    print('Finished Training')
    return np.asarray(history_MI)


# Add interpolated bounds
def sigmoid(x):
  return 1/(1. + np.exp(-x))


def sample_correlated_gaussian(rho=0.5, dim=20, data_size=10000):
    """Generate samples from a correlated Gaussian distribution."""
    x, eps = torch.split(torch.normal(0, 1, size=(data_size, 2 * dim)), dim, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho ** 2, dtype=torch.float32)) * eps
    return x, y


def rho_to_mi(dim, rho):
    return -0.5 * np.log(1 - rho ** 2) * dim


def mi_schedule(n_iter):
    mis = np.round(np.linspace(0.5, 5.5 - 1e-9, n_iter)) * 2.0  # 0.1
    return mis.astype(np.float32)


# critic_type = 'concat' # or 'separable'
# estimators = {
#     'NWJ': dict(estimator='nwj', critic=critic_type, baseline='constant'),
#     'TUBA': dict(estimator='tuba', critic=critic_type, baseline='unnormalized'),
#     'InfoNCE': dict(estimator='infonce', critic=critic_type, baseline='constant'),
# }
# for alpha_logit in [-5., 0., 5.]:
#     name = 'alpha=%.2f' % sigmoid(alpha_logit)
#     estimators[name] = dict(estimator='interpolated', critic=critic_type,
#                           alpha_logit=alpha_logit, baseline='unnormalized')


if __name__ == "__main__":
    pass
