# inference.py - Generate talking head images using trained model
import torch
from model import TalkingHeadGenerator
from torchvision.utils import save_image

def generate_talking_head():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TalkingHeadGenerator().to(device)
    generator.load_state_dict(torch.load("models/generator_epoch_90.pth", map_location=device))
    generator.eval()
    
    noise = torch.randn(1, 100).to(device)
    generated_image = generator(noise)
    save_image(generated_image, "output/generated_talking_head.png", normalize=True)
    print("Generated image saved as output/generated_talking_head.png")

if __name__ == "__main__":
    generate_talking_head()