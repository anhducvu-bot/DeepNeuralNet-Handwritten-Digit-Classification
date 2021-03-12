import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

epochs = 50
batch_size = 32

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST('MNIST_data/', download=True, train=True , transform=transform) #train: Define training set or test set
trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size, shuffle=True)

#Build a feed-forward network:
model = nn.Sequential(nn.Linear(784, 64), #784 neuron to 128 neuron
                      nn.ReLU(), #Activation function for the 128 neuron
                      nn.Linear(64,32), #128 neuron to 64 neuron
                      nn.ReLU(), #Activationf function for 64 neuron
                      nn.Linear(32,10))
                      #,nn.LogSoftmax(dim=1)) #64 neuron to 10 neuron

#Define the loss
criterion = nn.MSELoss()
#Optimizer to update weight with stochastic gradient descent:
optimizer = optim.SGD(model.parameters(),lr=0.01)


for e in range(epochs):
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1) #Flatten Input
        #Converting labels to correct dimensions:
        label_numpy = labels.numpy()
        labelz = np.zeros((batch_size,10))
        for i in range(len(label_numpy)):
            x = np.zeros(10)
            x[label_numpy[i]] = label_numpy[i]
            labelz[i] = x
        labelzz = torch.from_numpy(labelz).float()
        optimizer.zero_grad() #Zero out gradient after each iteration
        output = model.forward(images)
        loss = criterion(output,labelzz)
        loss.backward()
        optimizer.step()


#Testing
testset = datasets.MNIST('MNIST_data/', download=True, train=False , transform=transform) #train: Define training set or test set
testloader = torch.utils.data.DataLoader(testset,batch_size = 32, shuffle=True)

images, labels = next(iter(testloader))
random_image = images[np.random.randint(32)]
img = random_image.view(1,784)

with torch.no_grad(): logits = model.forward(img) #Not keeping track of the gradient to make the computation faster

print("The number:",logits.numpy().argmax())
#print("The regression for the number:", logits)
plt.imshow(random_image.numpy().squeeze())
plt.show()


