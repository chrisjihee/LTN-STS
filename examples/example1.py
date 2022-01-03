import ltn
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score

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
nr_samples = 100
nr_train = 50
batch_size = 64
dataset = torch.rand((nr_samples, 2))
labels_dataset = torch.sum(torch.square(dataset - torch.tensor([.5, .5])), dim=1) < .09

print()
print()
print()
print("""###
# LTN setting
###""")


# we define predicate A
class ModelA(torch.nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.layer1 = torch.nn.Linear(2, 16)
        self.layer2 = torch.nn.Linear(16, 16)
        self.layer3 = torch.nn.Linear(16, 1)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.elu(self.layer1(x))
        x = self.elu(self.layer2(x))
        x = self.layer3(x)
        o = self.sigmoid(x)
        return o


A = ltn.Predicate(ModelA()).to(ltn.device)

# we define the connectives, quantifiers, and the SatAgg
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
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
                 data,
                 labels,
                 batch_size=1,
                 shuffle=True):
        self.data = data.to(ltn.device)
        self.labels = labels
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
            data = self.data[idxlist[start_idx:end_idx]]
            labels = self.labels[idxlist[start_idx:end_idx]]

            yield data, labels


# define metrics for evaluation of the model

# it computes the overall satisfaction level on the knowledge base using the given data loader (train or test)
def compute_sat_level(loader):
    mean_sat = 0
    for data, labels in loader:
        x_A = ltn.Variable("x_A", data[torch.nonzero(labels)])  # positive examples
        x_not_A = ltn.Variable("x_not_A", data[torch.nonzero(torch.logical_not(labels))])  # negative examples
        mean_sat += SatAgg(
            Forall(x_A, A(x_A)),
            Forall(x_not_A, Not(A(x_not_A)))
        )
    mean_sat /= len(loader)
    return mean_sat


# it computes the overall accuracy of the predictions of the trained model using the given data loader
# (train or test)
def compute_accuracy(loader):
    mean_accuracy = 0.0
    for data, labels in loader:
        predictions = A.model(data).cpu().detach().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        mean_accuracy += accuracy_score(labels, predictions)

    return mean_accuracy / len(loader)


# create train and test loader, 50 points each
# batch size is 64, meaning there is only one batch for epoch
train_loader = DataLoader(dataset[:nr_train], labels_dataset[:nr_train], batch_size, True)
test_loader = DataLoader(dataset[nr_train:], labels_dataset[nr_train:], batch_size, False)

print()
print()
print()
print("""###
# Learning
###""")
optimizer = torch.optim.Adam(A.parameters(), lr=0.001)

# training of the predicate A using a loss containing the satisfaction level of the knowledge base
# the objective it to maximize the satisfaction level of the knowledge base
for epoch in range(1000):
    train_loss = 0.0
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        # we ground the variables with current batch data
        x_A = ltn.Variable("x_A", data[torch.nonzero(labels)])  # positive examples
        x_not_A = ltn.Variable("x_not_A", data[torch.nonzero(torch.logical_not(labels))])  # negative examples
        sat_agg = SatAgg(
            Forall(x_A, A(x_A)),
            Forall(x_not_A, Not(A(x_not_A)))
        )
        loss = 1. - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)

    # we print metrics every 20 epochs of training
    if epoch % 20 == 0:
        print(f" epoch {epoch} | loss {train_loss:.4f} |"
              f" Train Sat {compute_sat_level(train_loader):.3f} | Test Sat {compute_sat_level(test_loader):.3f} |"
              f" Train Acc {compute_accuracy(train_loader):.3f} | Test Acc {compute_accuracy(test_loader):.3f}")

fig = plt.figure(figsize=(9, 11))

plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax = plt.subplot2grid((3, 8), (0, 2), colspan=4)
ax.set_title("ground truth")
ax.scatter(dataset[labels_dataset][:, 0], dataset[labels_dataset][:, 1], label='A')
ax.scatter(dataset[torch.logical_not(labels_dataset)][:, 0], dataset[torch.logical_not(labels_dataset)][:, 1], label='~A')
ax.legend()

x_train = ltn.Variable("x", dataset[:nr_train])  # Training data
x_test = ltn.Variable("x", dataset[nr_train:])  # Test data

fig.add_subplot(3, 2, 3)
result = A(x_train)
plt.title("A(x_train)")
plt.scatter(dataset[:nr_train, 0], dataset[:nr_train, 1], c=result.value.cpu().detach().numpy().squeeze())
plt.colorbar()

fig.add_subplot(3, 2, 4)
result = Not(A(x_train))
plt.title("~A(x_train)")
plt.scatter(dataset[:nr_train, 0], dataset[:nr_train, 1], c=result.value.cpu().detach().numpy().squeeze())
plt.colorbar()

fig.add_subplot(3, 2, 5)
result = A(x_test)
plt.title("A(x_test)")
plt.scatter(dataset[nr_train:, 0], dataset[nr_train:, 1], c=result.value.cpu().detach().numpy().squeeze())
plt.colorbar()

fig.add_subplot(3, 2, 6)
result = Not(A(x_test))
plt.title("~A(x_test)")
plt.scatter(dataset[nr_train:, 0], dataset[nr_train:, 1], c=result.value.cpu().detach().numpy().squeeze())
plt.colorbar()

plt.savefig("example1.pdf")
plt.show()
