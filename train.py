import torch
from torch import nn, optim
from siren_pytorch import SirenNet, SirenWrapper
from dataset import *
from sampler import poisson_disk
from functions import *

"""
1. Train the model using train dataset
2. Do the validation process using test dataset

NOTE on how to calculate d(NN) / d(input):
    # test grad
    input_ = torch.tensor([[0.5, 0.5]], requires_grad=True)
    output_ = net(input_)
    output_.backward()
    print(output_)
    print(input_.grad)
"""
steps = 1000

net = SirenNet(
    dim_in=2,  # input dimension, ex. 2d coor
    dim_hidden=256,  # hidden dimension
    dim_out=1,  # output dimension, ex. rgb value
    num_layers=5,  # number of layers
    w0_initial=30.0,  # different signals may require different omega_0 in the first layer - this is a hyperparameter
)

samples = np.load("samples/6041-[[-5, 5], [-5, 5]].npy").astype(
    np.float32
)  # n x 2 list
gt = volcano_function(samples).reshape(-1, 1)

inputs = torch.tensor(samples)
gt = torch.tensor(gt)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

for i in range(steps):
    optimizer.zero_grad()
    eval = net(inputs)
    loss = mse_loss(eval, gt)
    print("step: ", i, "; MSE Loss:", loss.item())
    loss.backward()
    optimizer.step()

# net = torch.load("models/my_model_ackley")
output = net(inputs).detach().numpy().reshape(-1)

fig = plt.figure()
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection="3d")

ax0.set_xlabel("X")
ax0.set_ylabel("Y")
ax0.set_zlabel("Z")
ax0.set_title("Implicit Representation of volcano function")

ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("Analytical Ground Truth")

ax0.plot_trisurf(inputs[:, 0], inputs[:, 1], output, cmap="viridis")
ax1.plot_trisurf(inputs[:, 0], inputs[:, 1], gt.flatten(), cmap="viridis")
plt.show()


path = input("Input the model path: ")
print("save to: ", path)
torch.save(net, path)
