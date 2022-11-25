# import libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# XOR model
class XOR_NNModel(nn.Module):
    def __init__(self):
        super(XOR_NNModel, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
      x = self.linear(input)
      sig = self.Sigmoid(x)
      yh = self.linear2(sig)
      return yh

# Initalizing data
xInput = torch.Tensor([[0., 0.],
               [0., 1.],
               [1., 0.],
               [1., 1.]])
y = torch.Tensor([0., 1., 1., 0.]).reshape(xInput.shape[0], 1)

if __name__ == "__main__":
    # Initalizing model, setting hyper parameters
    xor_network = XOR_NNModel()
    epochs = 2000
    mseloss = nn.MSELoss()
    optimizer = torch.optim.Adam(xor_network.parameters(), lr = 0.03)
    all_losses = []
    current_loss = 0
    plot_every = 50
    for epoch in range(epochs):

        # input training example and return the prediction
        yhat = xor_network.forward(xInput)

        # calculate MSE loss
        loss = mseloss(yhat, y)

        # backpropogate through the loss gradiants
        loss.backward()

        # update model weights
        optimizer.step()

        # remove current gradients for next iteration
        optimizer.zero_grad()

        # append to loss
        current_loss += loss
        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

        # print progress
        if epoch % 500 == 0:
            print(f'Epoch: {epoch} completed')

    nploss = []
    for nums in all_losses:
        nploss.append(nums.detach().numpy())

    plt.plot(nploss)
    plt.ylabel('Loss')
    plt.savefig("xorgatepytorch.png")


    # show weights and bias
    for name, param in xor_network.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # test input
    input = torch.tensor([1., 0.])
    out = xor_network(input)
    print("Answer for [1, 0]: ",out.round())

    input = torch.tensor([0., 0.])
    out = xor_network(input)
    print("Answer for [0, 0]: ",out.round())