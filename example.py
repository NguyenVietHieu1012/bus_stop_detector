import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Hyperparameters
batch_size = 128
lr = 0.0002
nz = 100  # Dimension of latent vector
epochs = 50
image_size = 64
nc = 3  # Number of channels (for RGB images)

# Data loading
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize to consistent size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.RandomRotation(30),  # Random rotation within 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# Load your custom dataset
dataset_path = 'C:/Users/Admin/PycharmProjects/GamePro/bus_stop_detector/dataset'
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self, nz, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the generator and discriminator
netG = Generator(nz, nc).cuda()
netD = Discriminator(nc).cuda()

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # Update Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()
        real_data = data[0].cuda()
        b_size = real_data.size(0)
        label = torch.full((b_size, 1), 1., device='cuda')  # Adjusted shape
        output = netD(real_data).view(-1, 1)  # Ensure output has shape [batch_size, 1]
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device='cuda')
        fake_data = netG(noise)
        label.fill_(0.)
        output = netD(fake_data.detach()).view(-1, 1)  # Adjusted shape
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # Update Generator: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(1.)
        output = netD(fake_data).view(-1, 1)  # Adjusted shape
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 100 == 0:
            print(
                f'Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] Loss_D: {errD.item()}, Loss_G: {errG.item()} D(x): {D_x} D(G(z)): {D_G_z1}/{D_G_z2}')

# Save the generated images
fake_images = netG(torch.randn(64, nz, 1, 1, device='cuda')).detach().cpu()
torchvision.utils.save_image(fake_images, 'fake_images.png', normalize=True)