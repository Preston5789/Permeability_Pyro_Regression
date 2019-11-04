import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

# for CI testing
smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('0.4.1')
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

DATA_URL = "C:/Users/PrestonPhillips/Desktop/mapinv_reference_data_carbonates_calculatedMode.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

df = data[["POROSITY", "ROCK_INDEX","Mode","PERMEABILITY","DATA_SOURCE"]]
df = df[df["DATA_SOURCE"]=="Rosetta"]
df = data[["POROSITY", "ROCK_INDEX","Mode","PERMEABILITY"]]

df["PERMEABILITY"]= np.log(df["PERMEABILITY"])



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
rock_1_data = data[data["ROCK_INDEX"] == 1]
rock_4_data = data[data["ROCK_INDEX"] == 4]
sns.scatterplot(rock_4_data["POROSITY"],
            rock_4_data["PERMEABILITY"],
            ax=ax[0])
ax[0].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock1")
sns.scatterplot(rock_1_data["POROSITY"],
            rock_1_data["PERMEABILITY"],
            ax=ax[1])
ax[1].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock4")

plt.show()

class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        return self.linear(x) + (self.factor * x[:, 0] * x[:, 1] * x[:,2]).unsqueeze(1)

p = 3  # number of features
regression_model = RegressionModel(p)

loss_fn = torch.nn.MSELoss(reduction='sum')
optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)
num_iterations = 5000 if not smoke_test else 2
data = torch.tensor(df.values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]



def model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, 3), torch.ones(1, 3)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    scale = pyro.sample("sigma", Uniform(0., 10.))
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import init_to_feasible
guide = AutoDiagonalNormal(model, init_loc_fn=init_to_feasible)


optim = Adam({"lr": 0.02})
svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

def train():
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(data)))

train()

get_marginal = lambda traces, sites:EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()

def summary(traces, sites):
    marginal = get_marginal(traces, sites)
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))

posterior = svi.run(x_data, y_data)

# posterior predictive distribution we can get samples from
trace_pred = TracePredictive(wrapped_model,
                             posterior,
                             num_samples=1000)
post_pred = trace_pred.run(x_data, None)
post_summary = summary(post_pred, sites= ['prediction', 'obs'])
mu = post_summary["prediction"]
y = post_summary["obs"]
predictions = pd.DataFrame({
    "POROSITY": x_data[:, 0],
    "ROCK_INDEX": x_data[:, 1],
    "MODE": x_data[:,2],
    "mu_mean": mu["mean"],
    "mu_perc_5": mu["5%"],
    "mu_perc_95": mu["95%"],
    "y_mean": y["mean"],
    "y_perc_5": y["5%"],
    "y_perc_95": y["95%"],
    "true_Perm": y_data,
})




fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), sharey=True)
rock_1_data = predictions[predictions["ROCK_INDEX"] ==1]
rock_2_data = predictions[predictions["ROCK_INDEX"] ==2]
rock_3_data = predictions[predictions["ROCK_INDEX"] ==3]
rock_4_data = predictions[predictions["ROCK_INDEX"] == 4]
rock_1_data = rock_1_data.sort_values(by=["POROSITY"])
rock_2_data = rock_2_data.sort_values(by=["POROSITY"])
rock_3_data = rock_3_data.sort_values(by=["POROSITY"])
rock_4_data = rock_4_data.sort_values(by=["POROSITY"])
fig.suptitle("Regression line 90% CI", fontsize=16)
idx = np.argsort(rock_1_data["POROSITY"])
ax[0,0].plot(rock_1_data["POROSITY"],
           rock_1_data["mu_mean"])
ax[0,0].fill_between(rock_1_data["POROSITY"],
                   rock_1_data["mu_perc_5"],
                   rock_1_data["mu_perc_95"],
                   alpha=0.5)
ax[0,0].plot(rock_1_data["POROSITY"],
           rock_1_data["true_Perm"],
           "o")
ax[0,0].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock Type 1")
idx = np.argsort(rock_2_data["POROSITY"])
ax[0,1].plot(rock_2_data["POROSITY"],
           rock_2_data["mu_mean"])
ax[0,1].fill_between(rock_2_data["POROSITY"],
                   rock_2_data["mu_perc_5"],
                   rock_2_data["mu_perc_95"],
                   alpha=0.5)
ax[0,1].plot(rock_2_data["POROSITY"],
           rock_2_data["true_Perm"],
           "o")
ax[0,1].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock_Type 2")
idx = np.argsort(rock_3_data["POROSITY"])
ax[1,0].plot(rock_3_data["POROSITY"],
           rock_3_data["mu_mean"])
ax[1,0].fill_between(rock_3_data["POROSITY"],
                   rock_3_data["mu_perc_5"],
                   rock_3_data["mu_perc_95"],
                   alpha=0.5)
ax[1,0].plot(rock_3_data["POROSITY"],
           rock_3_data["true_Perm"],
           "o")
ax[1,0].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock_Type 3")
idx = np.argsort(rock_4_data["POROSITY"])
ax[1,1].plot(rock_4_data["POROSITY"],
           rock_4_data["mu_mean"])
ax[1,1].fill_between(rock_4_data["POROSITY"],
                   rock_4_data["mu_perc_5"],
                   rock_4_data["mu_perc_95"],
                   alpha=0.5)
ax[1,1].plot(rock_4_data["POROSITY"],
           rock_4_data["true_Perm"],
           "o")
ax[1,1].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock_Type 4")



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
fig.suptitle("Posterior predictive distribution with 90% CI", fontsize=16)
ax[0].plot(rock_4_data["POROSITY"],
           rock_4_data["y_mean"])
ax[0].fill_between(rock_4_data["POROSITY"],
                   rock_4_data["y_perc_5"],
                   rock_4_data["y_perc_95"],
                   alpha=0.5)
ax[0].plot(rock_4_data["POROSITY"],
           rock_4_data["true_Perm"],
           "o")
ax[0].set(xlabel="Pososity",
          ylabel="log Perm",
          title="Rock1")
idx = np.argsort(rock_1_data["POROSITY"])

ax[1].plot(rock_1_data["POROSITY"],
           rock_1_data["y_mean"])
ax[1].fill_between(rock_1_data["POROSITY"],
                   rock_1_data["y_perc_5"],
                   rock_1_data["y_perc_95"],
                   alpha=0.5)
ax[1].plot(rock_1_data["POROSITY"],
           rock_1_data["true_Perm"],
           "o")
ax[1].set(xlabel="POROSITY",
          ylabel="log Perm",
          title="Rock4")

# we need to prepend `module$$$` to all parameters of nn.Modules since
# that is how they are stored in the ParamStore
weight = get_marginal(posterior, ['module$$$linear.weight']).squeeze(1).squeeze(1)
factor = get_marginal(posterior, ['module$$$factor'])
gamma_within_africa = weight[:, 1] + factor.squeeze(1)
gamma_outside_africa = weight[:, 1]
fig = plt.figure(figsize=(10, 6))
sns.distplot(gamma_within_africa, kde_kws={"label": "Rock4"},)
sns.distplot(gamma_outside_africa, kde_kws={"label": "Rock1"})
fig.suptitle("Density of Slope : log(Perm) vs. POROSITY", fontsize=16)
plt.show()


from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
print(predictions.columns)

ax.scatter3D(rock_1_data["POROSITY"], rock_1_data["MODE"], rock_1_data["y_mean"], c=rock_1_data["y_mean"], cmap='Greens');

plt.show()
