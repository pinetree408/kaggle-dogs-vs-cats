import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 29 * 29, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DogsCatsDataSet(Dataset):
    def __init__(self, root_dir, train, transform=None):
        all_list = glob.glob(root_dir + "*.jpg")
        if train:
            self.img_list = all_list[:int(len(all_list)*0.9)]
        else:
            self.img_list = all_list[int(len(all_list)*0.9):]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        item = self.img_list[idx]
        image = io.imread(item)
        label = item.split('.')[0].split('/')[2]
        if label == 'cat':
            label = 0
        else:
            label = 1
        sample = {
            'image': image,
            'label': label
        }
        if self.transform:
            sample = {
                'image': self.transform(sample['image']),
                'label': label
            }
        return sample

def main():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = DogsCatsDataSet(
        root_dir='data/train/',
        train=True,
        transform=transform
    )
    test_dataset = DogsCatsDataSet(
        root_dir='data/train/',
        train=False,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=True
    )
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(1, 11):
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data['image'])
            loss = F.nll_loss(outputs, data['label'])
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
		print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, i * len(data['image']), len(train_loader.dataset), 100. * i / len(train_loader), loss.item())
        model.eval()
        test_loss = 0
	correct = 0
	with torch.no_grad():
            for data in test_loader:
		outputs = model(data['image'])
                test_loss += F.nll_loss(outputs, data['label'], reduction='sum').item() # sum up batch loss
                pred = outputs.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(data['label'].view_as(pred)).sum().item()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    main()
