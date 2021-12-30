import torch

import ltn

print()
print()
print()
print("""###
# Constants
###""")
c1 = ltn.Constant(torch.tensor([2.1, 3]))
c2 = ltn.Constant(torch.tensor([[4.2, 3, 2.5], [4, -1.3, 1.8]]))
c3 = ltn.Constant(torch.tensor([0., 0.]), trainable=True)
print(c1.value)
print(c2.value)
print(c3.value)
# here, we have to perform a detach before calling numpy(), because the tensor has requires_grad=True
print(c3.value.cpu().detach().numpy())

print()
print()
print()
print("""###
# Predicates
###""")
mu = ltn.Constant(torch.tensor([2., 3.]))
P1 = ltn.Predicate(func=lambda x: torch.exp(-torch.norm(x - mu.value, dim=1)))


class ModelP2(torch.nn.Module):
    def __init__(self):
        super(ModelP2, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dense1 = torch.nn.Linear(2, 5)
        self.dense2 = torch.nn.Linear(5, 1)  # returns one value in [0,1]

    def forward(self, x):
        x = self.elu(self.dense1(x))
        return self.sigmoid(self.dense2(x))


modelP2 = ModelP2()
P2 = ltn.Predicate(model=modelP2)
c1 = ltn.Constant(torch.tensor([2.1, 3]))
c2 = ltn.Constant(torch.tensor([4.5, 0.8]))
c3 = ltn.Constant(torch.tensor([3.0, 4.8]))
print(P1(c1).value)
print(P1(c2).value)
print(P2(c3).value)
print(P1(c1))
print()


class ModelP4(torch.nn.Module):
    def __init__(self):
        super(ModelP4, self).__init__()
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dense1 = torch.nn.Linear(4, 5)
        self.dense2 = torch.nn.Linear(5, 1)  # returns one value in [0,1]

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.elu(self.dense1(x))
        return self.sigmoid(self.dense2(x))


P4 = ltn.Predicate(ModelP4())
c1 = ltn.Constant(torch.tensor([2.1, 3]))
c2 = ltn.Constant(torch.tensor([4.5, 0.8]))
print(P4(c1, c2).value)  # multiple arguments are passed as a list

print()
print()
print()
print("""###
# Functions
###""")
f1 = ltn.Function(func=lambda x, y: x - y)


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = torch.nn.Linear(2, 10)
        self.dense2 = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        return self.dense2(x)


model_f2 = MyModel()
f2 = ltn.Function(model=model_f2)

c1 = ltn.Constant(torch.tensor([2.1, 3]))
c2 = ltn.Constant(torch.tensor([4.5, 0.8]))
print(f1(c1, c2).value)  # multiple arguments are passed as a list
print(f2(c1).value)
print(f2(c1))

print()
print()
print()
print("""###
# Variables
###""")
x = ltn.Variable('x', torch.randn((10, 2)))
y = ltn.Variable('y', torch.randn((5, 2)))

# Notice that the outcome is a 2-dimensional tensor where each cell
# represents the satisfiability of P4 evaluated with each individual in x and in y.
P4 = ltn.Predicate(ModelP4())
res1 = P4(x, y)
print(res1.shape())
print(res1.free_vars)  # dynamically added attribute; tells that axis 0 corresponds to x and axis 1 to y
print(res1.value[2, 0])  # gives the result computed with the 3rd individual in x and the 1st individual in y
print(res1.value)
print()

# Notice that the last axe(s) correspond to the dimensions of the outcome;
# here, f2 projects to outcomes in R^2, so the outcome has one additional axis of dimension 2.
# the output tensor has shape (10, 5, 2) because variable x has 10 individuals, y has 5 individuals, and f1 maps in R^2
res2 = f1(x, y)
print(res2.shape())
print(res2.free_vars)
print(res2.value[2, 0])  # gives the result calculated with the 3rd individual in x and the 1st individual in y
print(res2.value)
print()

c1 = ltn.Constant(torch.tensor([2.1, 3]))
res3 = P4(c1, y)
print(res3.shape())  # Notice that no axis is associated to a constant. The output has shape (5,) because variable y has
# 5 individuals
print(res3.free_vars)
print(res3.value[0])  # gives the result calculated with c1 and the 1st individual in y
print(res3.value)

print()
print()
print()
print("""###
# Variables made of trainable constants
###""")
c1 = ltn.Constant(torch.tensor([2.1, 3]), trainable=True)
c2 = ltn.Constant(torch.tensor([4.5, 0.8]), trainable=True)

# PyTorch will keep track of the gradients between c1, c2 and x.
# Read tutorial 3 for more details.
x = ltn.Variable('x', torch.stack([c1.value, c2.value]))
res = P2(x)
print(res.value)

print(x.value[0])
print(x.value[1])

print(c1.value.grad_fn)
print(c2.value.grad_fn)
