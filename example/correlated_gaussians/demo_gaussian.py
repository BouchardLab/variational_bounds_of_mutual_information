#!/usr/bin/env python

"""demo_gaussian.py: Showcase of how to use mutual information as an estimator. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append("../../")
from Mi_estimator import train_estimator, rho_to_mi, sample_correlated_gaussian, Mi_estimator


class JointData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        return sample


if __name__ == "__main__":
    critic_type = 'concat'  # or 'separable'
    estimators = {
        'NWJ': dict(estimator='nwj', critic=critic_type, baseline='constant'),
        'TUBA': dict(estimator='tuba', critic=critic_type, baseline='unnormalized'),
        'InfoNCE': dict(estimator='infonce', critic=critic_type, baseline='constant'),
    }
    data_params = {
        'dim': 20,
        'batch_size': 256,
        'rho': 0.7,
        'data_size': 10000
    }
    critic_params = {
        'n_layers': 2,
        'x_dim': 20,
        'embed_dim': 256,
        'y_dim': 20,
        'activation': 'relu',
    }
    opt_params = {
        'iterations': 20000,
        'n_epochs': 100,
        'learning_rate': 5e-4,
    }

    # generate data
    x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=data_params['rho'],
                                      data_size=data_params['data_size'])
    xy_dataset = JointData(x, y)
    dataloader = DataLoader(xy_dataset, batch_size=data_params["batch_size"], shuffle=True)

    estimates = {}
    for estimator, mi_params in estimators.items():
        print("Training %s..." % estimator)
        mi_est = Mi_estimator(critic_params, data_params, mi_params, opt_params)
        estimates[estimator] = mi_est.fit(dataloader, epochs=opt_params["n_epochs"])

    # Smooting span for Exponential Moving Average
    EMA_SPAN = 20
    # Ground truth MI
    mi_true = rho_to_mi(data_params['dim'], data_params['rho'])

    nrows = min(1, len(estimates))
    ncols = int(np.ceil(len(estimates) / float(nrows)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3 * nrows))
    if len(estimates) == 1:
        axs = [axs]
    axs = np.ravel(axs)
    names = np.sort(list(estimators.keys()))
    for i, name in enumerate(names):
        plt.sca(axs[i])
        # Plot estimated MI and smoothed MI
        mis = estimates[name]
        mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
        p1 = plt.plot(mis, alpha=0.3)[0]
        plt.plot(mis_smooth, c=p1.get_color())
        plt.title("{}, true MI is {}".format(name, np.round(mi_true, 2)))
        plt.xlabel('steps')
        if i % ncols == 0:
            plt.ylabel('Mutual information')
    plt.legend(loc='best', fontsize=8, framealpha=0.0)
    plt.gcf().tight_layout()
    plt.show()

    import pdb; pdb.set_trace()