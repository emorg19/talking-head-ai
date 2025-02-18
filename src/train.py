# train.py - Training script
import torch
from torch.utils.data import DataLoader
from model import TalkingHeadGenerator, TalkingHeadDiscriminator
from dataset import TalkingHeadDataset
import torch.nn as nn
import os
from torchvision.utils import save_image

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TalkingHeadGenerator().to(device)
    discriminator = TalkingHeadDiscriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    dataset = TalkingHeadDataset("data/video_frames", "data/audio_files")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(100):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()
        
        print(f"Epoch [{epoch+1}/100] Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")
        if epoch % 10 == 0:
            save_image(fake_images[:25], f"output/epoch_{epoch}.png", nrow=5, normalize=True)
            torch.save(generator.state_dict(), f"models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"models/discriminator_epoch_{epoch}.pth")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    train_model()
