import matplotlib.pyplot as plt
import numpy as np
import torch

import ltn

print()
print()
print()
print("""###
# Classification with Nearest Neighbour
###""")
points = np.array(
    [[0.4, 0.3], [1.2, 0.3], [2.2, 1.3], [1.7, 1.0], [0.5, 0.5], [0.3, 1.5], [1.3, 1.1], [0.9, 1.7],
     [3.4, 3.3], [3.2, 3.3], [3.2, 2.3], [2.7, 2.0], [3.5, 3.5], [3.3, 2.5], [3.3, 1.1], [1.9, 3.7], [1.3, 3.5],
     [3.3, 1.1], [3.9, 3.7]])
point_a = [3.3, 2.5]
point_b = [1.3, 1.1]
fig, ax = plt.subplots()
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.scatter(points[:, 0], points[:, 1], color="black", label="unknown")
ax.scatter(point_a[0], point_a[1], color="blue", label="a")
ax.scatter(point_b[0], point_b[1], color="red", label="b")
ax.set_title("Dataset of individuals")
plt.legend()
plt.show()

print()
print()
print()
print("""###
# Definition of knowledge base
###""")


# Predicate C
class ModelC(torch.nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.dense1 = torch.nn.Linear(2, 5)
        self.dense2 = torch.nn.Linear(5, 5)
        self.dense3 = torch.nn.Linear(5, 2)

    def forward(self, x, l):
        """x: point, l: one-hot label"""
        x = self.elu(self.dense1(x))
        x = self.elu(self.dense2(x))
        prob = self.softmax(self.dense3(x))
        return torch.sum(prob * l, dim=1)


C = ltn.Predicate(ModelC())

# Predicate Sim
Sim = ltn.Predicate(func=lambda u, v: torch.exp(-1. * torch.sqrt(torch.sum(torch.square(u - v), dim=1))))

# variables and constants
x1 = ltn.Variable("x1", torch.tensor(points))
x2 = ltn.Variable("x2", torch.tensor(points))
a = ltn.Constant(torch.tensor([3.3, 2.5]))
b = ltn.Constant(torch.tensor([1.3, 1.1]))
l_a = ltn.Constant(torch.tensor([1, 0]))
l_b = ltn.Constant(torch.tensor([0, 1]))
l = ltn.Variable("l", torch.tensor([[1, 0], [0, 1]]))

similarities_to_a = Sim(x1, a).value
fig, ax = plt.subplots()
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.scatter(points[:, 0], points[:, 1], color="black")
point_a = a.value.cpu().detach().numpy()
ax.scatter(point_a[0], point_a[1], color="blue")
ax.set_title("Illustrating the similarities of each point to a")
for i, sim_to_a in enumerate(similarities_to_a):
    s = sim_to_a.cpu().detach().numpy()
    plt.plot([points[i, 0], point_a[0]], [points[i, 1], point_a[1]],
             alpha=float(s), color="blue")
plt.show()

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")

# by default, SatAgg uses the pMeanError
sat_agg = ltn.fuzzy_ops.SatAgg()

# we need to learn the parameters of the predicate C
optimizer = torch.optim.Adam(C.parameters(), lr=0.001)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = 1. - sat_agg(
        C(a, l_a),
        C(b, l_b),
        Forall([x1, x2, l], Implies(Sim(x1, x2), Equiv(C(x1, l), C(x2, l))))
    )
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print("Epoch %d: Sat Level %.3f " % (epoch, 1 - loss.item()))
print("Training finished at Epoch %d with Sat Level %.3f" % (epoch, 1 - loss.item()))

fig = plt.figure(figsize=(10, 3))
fig.add_subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c=C(x1, l_a).value.cpu().detach().numpy(), vmin=0, vmax=1)
plt.title("C(x,l_a)")
plt.colorbar()
fig.add_subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 1], c=C(x1, l_b).value.cpu().detach().numpy(), vmin=0, vmax=1)
plt.title("C(x,l_b)")
plt.colorbar()
plt.show()

print()
print()
print()
print("""###
# Special Cases
###""")
r1 = 0
r2 = 4
points = (r1 - r2) * torch.rand((10000, 2)) + r2
points[-1] = torch.tensor([3., 3.])
points[-2] = torch.tensor([1., 1.])
points_a = torch.tensor([3., 3.])
points_b = torch.tensor([1., 1.])
a = ltn.Constant(torch.tensor([3., 3.]))
b = ltn.Constant(torch.tensor([1., 1.]))

fig, ax = plt.subplots()
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.scatter(points[:, 0], points[:, 1], color="black", label="unknown")
ax.scatter(point_a[0], point_a[1], color="blue", label="a")
ax.scatter(point_b[0], point_b[1], color="red", label="b")
ax.set_title("Dataset of individuals")
plt.legend()
plt.show()

# we define C again to re-initialize its weights
C = ltn.Predicate(ModelC())


# data loader which creates the batches
class DataLoader:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=True):
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            batch_points = self.data[idxlist[start_idx:end_idx]]

            yield batch_points


train_loader = DataLoader(points, 512)

# by default, SatAgg uses the pMeanError
sat_agg = ltn.fuzzy_ops.SatAgg()

# we need to learn the parameters of the predicate C
optimizer = torch.optim.Adam(C.parameters(), lr=0.001)

for epoch in range(100):
    for (batch_idx, (batch_points)) in enumerate(train_loader):
        x1 = ltn.Variable("x1", batch_points)
        x2 = ltn.Variable("x2", batch_points)
        optimizer.zero_grad()
        loss = 1. - sat_agg(
            C(a, l_a),
            C(b, l_b),
            Forall([x1, x2, l], Implies(Sim(x1, x2), Equiv(C(x1, l), C(x2, l))))
        )
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("Epoch %d: Sat Level %.3f " % (epoch, 1 - loss.item()))
print("Training finished at Epoch %d with Sat Level %.3f" % (epoch, 1 - loss.item()))

x1 = ltn.Variable("x1", points)
x2 = ltn.Variable("x2", points)
fig = plt.figure(figsize=(10, 3))
fig.add_subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], c=C(x1, l_a).value.cpu().detach().numpy(), vmin=0, vmax=1)
plt.title("C(x,l_a)")
plt.colorbar()
fig.add_subplot(1, 2, 2)
plt.scatter(points[:, 0], points[:, 1], c=C(x1, l_b).value.cpu().detach().numpy(), vmin=0, vmax=1)
plt.title("C(x,l_b)")
plt.colorbar()
plt.show()
