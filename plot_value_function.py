import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import sys
import argparse
import torch
from itertools import permutations
from all.environments.state import State
from all.approximation.v_network import VModule
from allagents.models import *

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a critics learned weights and plot the state space")
    parser.add_argument("weights", nargs='+', help="weights to load")
    parser.add_argument("--rbf", action='store_true', help="use rbf features")
    args = parser.parse_args()

    for weights in args.weights:
        model = torch.load(weights)

        r = np.linspace(-1, 1, 61)
        perms = list(permutations(r, 2))
        # print(perms[0:10, :])
        perms = torch.as_tensor(perms, device="cuda", dtype=torch.float32)

        if args.rbf:
            features = RBFKernel([[-1.0, 1.0], [-1.0, 1.0]], 41, 0.3)
            perms_featurised = features(perms)
            states = State(perms_featurised)
        else:
            states = State(perms)
        
        with torch.no_grad():
            values = model(states)
        
        if values.ndim > 1:
            values = torch.flatten(values)

        dt = np.dtype('float, float')
        perms = perms.cpu().numpy()

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_trisurf(perms[:, 0], perms[:, 1], values.cpu().numpy(), cmap=cm.jet, linewidth=0.2)
        # ax.plot_trisurf(perms[:, 0], perms[:, 1], np.ones(perms[:, 0].shape), cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel("Angle", fontsize=30, labelpad=20)
        ax.set_ylabel("Angle Dt", fontsize=30, labelpad=20)
        ax.set_zlabel("Output Value", fontsize=30, labelpad=20)
        ax.set_title(weights)

    plt.show()
