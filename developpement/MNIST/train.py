import argparse
import os
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import MNISTNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, optimizer, loader, writer, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)


def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='MNIST', help='experiment name')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST('data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.MNIST('data', download=True, train=False, transform=transform)

    # num_workers=0 : évite les soucis de mémoire partagée dans Docker
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = MNISTNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    writer = SummaryWriter(f'runs/{args.exp_name}')

    train(net, optimizer, trainloader, writer, epochs=args.epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy: {test_acc}')

    os.makedirs('weights', exist_ok=True)
    torch.save(net.state_dict(), 'weights/mnist_net.pth')
    print('Weights saved to weights/mnist_net.pth')