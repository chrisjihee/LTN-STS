import ltn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

# set cuda device
gpu_id = 1
gpu_ids = tuple([n for n in range(torch.cuda.device_count())])
cuda_ids = tuple([f"cuda:{n}" for n in gpu_ids])
device = torch.device(cuda_ids[gpu_id] if gpu_id in gpu_ids and torch.cuda.is_available() else "cpu")
ltn.device = device
print("\n" + "=" * 112)
print(f"[device] {device} ∈ [{', '.join(cuda_ids)}]")
print("=" * 112 + "\n")

print()
print()
print()
print("""###
# Dataset
###""")
data = pd.read_csv("datasets/real-estate.csv")
data = data.sample(frac=1)  # shuffle

x = torch.tensor(data[['X1 transaction date', 'X2 house age',
                       'X3 distance to the nearest MRT station',
                       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']].to_numpy()).float()
y = torch.tensor(data[['Y house price of unit area']].to_numpy()).float()
x_train, y_train = x[:330], y[:330]
x_test, y_test = x[330:], y[330:]

print()
print()
print()
print("""###
# LTN setting
###""")


# we define function f
class MLP(torch.nn.Module):
    """
    This model returns the prediction of the price of an house given in input. The output is linear since we are applying
    the model to a regression problem.
    """

    def __init__(self, layer_sizes=(6, 8, 8, 1)):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, x):
        """
        Method which defines the forward phase of the neural network for our regression task.

        :param x: the features of the example
        :return: prediction for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
        out = self.linear_layers[-1](x)
        return out


f = ltn.Function(MLP()).to(ltn.device)

# Equality Predicate - not trainable
alpha = 0.05
Eq = ltn.Predicate(func=lambda u, v: torch.exp(-alpha * torch.sqrt(torch.sum(torch.square(u - v), dim=1)))).to(ltn.device)

# we define the universal quantifier and the SatAgg operator
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
SatAgg = ltn.fuzzy_ops.SatAgg()

print()
print()
print()
print("""###
# Utils
###""")


# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 x,
                 y,
                 batch_size=1,
                 shuffle=True):
        self.x = x.to(ltn.device)
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.x.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            x = self.x[idxlist[start_idx:end_idx]]
            y = self.y[idxlist[start_idx:end_idx]]

            yield x, y


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for x_data, y_data in loader:
        x = ltn.Variable("x", x_data)
        y = ltn.Variable("y", y_data)
        mean_sat += Forall(ltn.diag(x, y), Eq(f(x), y)).value
    mean_sat /= len(loader)
    return mean_sat


# it computes the overall RMSE between the predictions and the ground truth, using the given data loader (train or test)
def compute_rmse(loader):
    mean_rmse = 0.0
    for x, y in loader:
        predictions = f.model(x).cpu().detach().numpy()
        mean_rmse += mean_squared_error(y, predictions, squared=False)
    return mean_rmse / len(loader)


# create train and test loader
train_loader = DataLoader(x_train, y_train, 64, shuffle=True)
test_loader = DataLoader(x_test, y_test, 64, shuffle=False)

print()
print()
print()
print("""###
# Learning
###""")
optimizer = torch.optim.Adam(f.parameters(), lr=0.0005)

for epoch in range(500):
    train_loss = 0.0
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        optimizer.zero_grad()
        # ground the variables with current batch of data
        x = ltn.Variable("x", x_data)  # samples
        y = ltn.Variable("y", y_data)  # ground truths
        sat_agg = Forall(ltn.diag(x, y), Eq(f(x), y)).value
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 50 epochs of training
    if epoch % 50 == 0:
        print(" epoch %d | loss %.4f | Train Sat %.3f | Test Sat %.3f | Train RMSE %.3f | Test RMSE %.3f " %
              (epoch, train_loss, compute_sat_level(train_loader), compute_sat_level(test_loader),
               compute_rmse(train_loader), compute_rmse(test_loader)))
