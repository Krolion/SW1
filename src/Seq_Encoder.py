from __future__ import print_function
import argparse
import torch
import os
import numpy as np
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
model.load_state_dict(torch.load('./models/last_model' ))
model.eval()
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


if __name__ == "__main__":
    PATH = 'sequences/1'
    global_output_data = np.array([])
    for j in range(500):
        LPATH = PATH + '/seq_' + str(j)
        with open(LPATH + '/actions.txt', 'r') as f:
            actions = f.read()
        local_data = []
        output_data = np.array([])
        for i in range(150):
            local_data.append(image_loader(LPATH + '/' + str(i) + '.png'))
            encoded = model.encode(local_data[i].view(-1,6144))[0][0]
            no_grad = model.encode(local_data[i].view(-1,6144))[0][0].detach()
            output_data = np.append(output_data, no_grad)
        seq = torch.from_numpy(output_data).view(150, 100)
        global_output_data = np.append(global_output_data, seq)
        torch.save(seq, LPATH + '/encoded.txt')
        print("Sequence ", str(j), " finished")
    torch.save(global_output_data, PATH + '/encoded.txt')



