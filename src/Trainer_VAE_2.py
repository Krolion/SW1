from __future__ import print_function
import argparse
import torch
import random
import os
import torch.utils.data
from torch import nn, optim, save
from PIL import Image
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 256 if torch.cuda.is_available() else 64

loader = transforms.Compose([
    transforms.Resize(imsize), 
    transforms.ToTensor()])  


def image_loader(image_name):
    image = Image.open(image_name).convert('L')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class MyDataset(Dataset):

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = [(image_loader("./sequences/1/seq_" + str(random.randint(0,499)) + "/" + str(random.randint(0,149)) + ".png"), torch.tensor(i))  for i in range(3600)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


FU_train = MyDataset()
FU_test = MyDataset()

print(len(FU_train))Во сколько?
print(FU_train[0][0][0].size())
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(FU_train, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(FU_test, batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(6144, 400)
        self.fc21 = nn.Linear(400, 100)
        self.fc22 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 400)
        self.fc4 = nn.Linear(400, 6144)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 6144))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
try:
    model.load_state_dict(torch.load('./models/last_model' ))
    model.eval()
except:
    pass
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 6144), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    ax = 0
    print(len(train_loader.dataset)," ",len(train_loader))
    for batch_idx, (data,_)  in enumerate(train_loader):
        ax += 1
        print(data.size())
        data.resize_((args.batch_size, 1, 64, 96))
        print(data.size())
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),	
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 20)
                data.resize_((args.batch_size, 1, 64, 96))
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 64, 96)[:n]])
                save_image(comparison.cpu(), '../bin/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        if epoch == args.epochs:
            torch.save(model.state_dict(), './models/last_model' )
        with torch.no_grad():
            sample = torch.randn(4, 100).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(4, 1, 64, 96), '../bin/results/sample_' + str(epoch) + '.png')