import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import CIFAR10Net
from albumentations_transform import AlbumentationsTransform

class CIFAR10Dataset(Dataset):
    def __init__(self, data, targets, transform=None, train=True):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.train = train

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img, self.train)
            
        return img, target

    def __len__(self):
        return len(self.data)

def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(f'Epoch: {epoch} Loss: {loss.item():.4f} Acc: {100*correct/processed:.2f}%')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return accuracy

def main():
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(SEED)
    
    # Get transforms
    transform = AlbumentationsTransform()
    
    # Load CIFAR10
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Create datasets
    train_dataset = CIFAR10Dataset(trainset.data, trainset.targets, transform, train=True)
    test_dataset = CIFAR10Dataset(testset.data, testset.targets, transform, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model, criterion, optimizer
    model = CIFAR10Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # Training loop
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, DEVICE, train_loader, optimizer, criterion, epoch)
        acc = test(model, DEVICE, test_loader, criterion)
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main()