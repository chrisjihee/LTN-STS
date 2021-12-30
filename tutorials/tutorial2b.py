import torch

import ltn

print()
print()
print()
print("""###
# Querying
###""")
x1 = torch.tensor(0.4)
x2 = torch.tensor(0.7)

# the stable keyword is explained at the end of the notebook
and_prod = ltn.fuzzy_ops.AndProd(stable=False)
and_luk = ltn.fuzzy_ops.AndLuk()

print(and_prod(x1, x2))
print(and_luk(x1, x2))

xs = torch.tensor([1., 1., 1., 0.5, 0.3, 0.2, 0.2, 0.1])

# the stable keyword is explained at the end of the notebook
forall_min = ltn.fuzzy_ops.AggregMin()
forall_pME = ltn.fuzzy_ops.AggregPMeanError(p=4, stable=False)

print(forall_min(xs, dim=0))
print(forall_pME(xs, dim=0))

print()
print()
print()
print("""###
# Learning
###""")
x1 = torch.tensor(0.3, requires_grad=True)
x2 = torch.tensor(0.5, requires_grad=True)
y = and_luk(x1, x2)
y.backward()  # this is necessary to compute the gradients
res = y.item()
gradients = [v.grad for v in [x1, x2]]
print(res)  # print the result of the aggregation
print(gradients)  # print gradients of x1 and x2

xs = torch.tensor([1., 1., 1., 0.5, 0.3, 0.2, 0.2, 0.1], requires_grad=True)
y = forall_min(xs, dim=0)
y.backward()
res = y.item()
gradients = xs.grad
print(res)  # print the result of the aggregation
print(gradients)  # print gradients of xs

xs = torch.tensor([1., 1., 1.], requires_grad=True)
y = forall_pME(xs, dim=0, p=4)
y.backward()
res = y.item()
gradients = xs.grad
print(res)  # print the result of the aggregation
print(gradients)  # print the gradients of xs

print()
print()
print()
print("""###
# Stable Product Configuration
###""")
xs = torch.tensor([1., 1., 1.], requires_grad=True)
y = forall_pME(xs, dim=0, p=4, stable=True)  # the exploding gradient problem is solved
y.backward()
res = y.item()
gradients = xs.grad
print(res)  # print the result of the aggregation
print(gradients)  # print the gradients of xs

xs = torch.tensor([1., 1., 1., 0.5, 0.3, 0.2, 0.2, 0.1], requires_grad=True)
y = forall_pME(xs, dim=0, p=4)
y.backward()
res = y.item()
gradients = xs.grad
print(res)  # print result of aggregation
print(gradients)  # print gradients of xs

xs = torch.tensor([1., 1., 1., 0.5, 0.3, 0.2, 0.2, 0.1], requires_grad=True)
y = forall_pME(xs, dim=0, p=20)
y.backward()
res = y.item()
gradients = xs.grad
print(res)  # print result of aggregation
print(gradients)  # print gradients of xs
