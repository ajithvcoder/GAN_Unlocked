import numpy as np
import matplotlib.pyplot as plt

class XOR_NNModel(object):
  def __init__(self):
    self.inputLayerSize = 2 # x1 and x2
    self.outputLayerSizer = 1 # y1
    self.hiddenLayeSizer = 4
    self.w1 = np.random.randn(self.inputLayerSize,self.hiddenLayeSizer)
    self.w2 = np.random.randn(self.hiddenLayeSizer,self.outputLayerSizer)

  def feedForward(self,x):
    self.z = np.dot(x,self.w1) # product of x input
    self.z2 = self.activationSigmoid(self.z)
    self.z3 = np.dot(self.z2,self.w2)
    o = self.activationSigmoid(self.z3)
    return o

  def backwardPropagate(self , x , y,o):
    self.o_error = y - o # calculate error in output
    self.o_delta = self.o_error * self.activationSigmoidPrime(o)
    self.z2_error = self.o_delta.dot(self.w2.T)
    self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)
    self.w1 +=x.T.dot(self.z2_delta)
    self.w2 += self.z2.T.dot(self.o_delta)

  def activationSigmoid(seld,s):
    return 1/(1+np.exp(-s))

  def activationSigmoidPrime(self,s):
    return s*(1-s)

  def perdictOutput(self, cInput):
    print("Predicted XOR output data based on trained weights:")
    print("Answer for ",str(cInput),"-", str(self.feedForward(cInput)))

# Initialze data
xInput = np.array(([0,0],[0,1],[1,0],[1,1]),dtype=float)
xInput = xInput/np.amax(xInput,axis=0)
y = np.array(([0],[1],[1],[0]),dtype=float)

if __name__ == "__main__":

    # Initalize model and set hyperparameters
    xor_network = XOR_NNModel()
    trainingEpochs = 2000
    nploss = []
    current_loss = 0
    plot_every = 50

    for i in range(trainingEpochs):
        out = xor_network.feedForward(xInput)
        xor_network.backwardPropagate(xInput, y, out)
        #Todo: verify if mean square loss is used in model
        loss = np.mean(np.square(y - xor_network.feedForward(xInput)))

        # append to loss
        current_loss += loss
        if i % plot_every == 0:
            nploss.append(current_loss / plot_every)
            current_loss = 0

        if i% 500 == 0:
            print("Epochs #" + str(i))
            print("sum squared loss : \n" + str(loss))

    plt.plot(nploss)
    plt.ylabel('Loss')
    plt.savefig("xorgatepython.png")

    # Test custom inputs
    customInput = np.array((1,0),dtype=float)
    xor_network.perdictOutput(customInput)

    customInput = np.array((0,0),dtype=float)
    xor_network.perdictOutput(customInput)
