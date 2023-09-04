import torch 
import torchvision 
import torchvision.transforms as transforms 
import torchvision.models as models
import torch.optim as optim 
import torch.nn as nn

transform = transforms.Compose( 
[transforms.ToTensor(), 
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

batch_size = 128 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
download=True, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
shuffle=False, num_workers=2) 

classes = ('plane', 'car', 'bird', 'cat', 
'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

net = models.resnet18()

criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) 

cnt = 0
for epoch in range(5): 
    for i, (inputs, labels) in enumerate(trainloader, 0):

        pred = net(inputs)
        loss = criterion(pred, labels)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        cnt = cnt + 1
        
        if (cnt % 1000 == 0):
            print(cnt ,'. ' , "Loss = " , loss.item())

correct = 0 
total = 0 

with torch.no_grad():
    for images, labels in testloader:
        pred = net(images)
        
        _, predicted = torch.max(pred, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 
100 * correct / total))