import torch
from PIL import Image
from loader import loader
from model import ImageCaptioningModel
from torchvision import transforms
import os
import argparse

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_loader, val_loader, full_dataset  = loader(
    root_folder="flickr30k_images/flickr30k_images/",
    desc_file="flickr30k_images/results.csv",
    transform=transform,
    batchsize=16,
    worker=8
)

embed_size = 256
hidden_size = 256
vocab_size = len(full_dataset.vocab)
num_layers = 1

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join('model_latest.pth')
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def predict_image_caption(image_path, model, vocab, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)
    with torch.no_grad():
        caption = model.caption(image_tensor, vocab)
    return " ".join(caption)

if __name__ == "__main__":
    caption = predict_image_caption(args.image, model, full_dataset.vocab, transform, device)
    print("📷 Caption générée :", caption)
