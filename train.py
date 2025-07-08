import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from loader import loader
from model import ImageCaptioningModel

import nltk
from nltk.translate.bleu_score import corpus_bleu

def evaluate_bleu(model, data_loader, vocab, device, max_samples=100):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for idx, (images, captions) in enumerate(data_loader):
            images = images.to(device)

            for i in range(images.size(0)):
                image = images[i].unsqueeze(0).to(device)

                gt_caption = captions[i]

                # Référence : transformer l’index en mots
                reference = []
                for idx_token in gt_caption:
                    word = vocab.idx2word[idx_token.item()]
                    if word == "<PAD>":
                        break
                    reference.append(word)
                references.append([reference])  # format attendu : list of list

                # Prédiction du modèle
                predicted = model.caption(image.to(device), vocab)

                hypotheses.append(predicted)

                if len(hypotheses) >= max_samples:
                    break
            if len(hypotheses) >= max_samples:
                break

    bleu4 = corpus_bleu(references, hypotheses)
    model.train()
    return bleu4


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Hyperparamètres après chargement du dataset
    embed_size = 256
    hidden_size = 256
    vocab_size = len(full_dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
    model.to(device)

    pad_idx = full_dataset.vocab.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=full_dataset.vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    writer = SummaryWriter(log_dir="runs/image_captioning")
    sample_images, sample_captions = next(iter(train_loader))
    writer.add_graph(model, input_to_model=(sample_images, sample_captions))

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        
        for idx, (images, captions) in enumerate(tqdm(train_loader)):

            images, captions = images.to(device), captions.to(device)
            
            lengths = []
            max_len = captions.size(1)

            for caption in captions:
                pad_positions = (caption == pad_idx).nonzero(as_tuple=True)[0]
                if len(pad_positions) == 0:
                    length = max_len
                else:
                    length = pad_positions[0].item()
                if length == max_len:
                    length = max_len - 1
                lengths.append(length)

            outputs = model(images, captions, lengths)      # (batch_size, seq_len+1, vocab_size)
            outputs = outputs[:, 1:, :]                      # enlever la première prédiction liée à l'image
            outputs = outputs.reshape(-1, outputs.shape[2]) # aplatir pour CrossEntropyLoss
            targets = captions[:, 1:].reshape(-1)           # aplatir les vraies étiquettes

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            if idx % 100 == 0:
                print(f"Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                global_step = epoch * len(train_loader) + idx
                writer.add_scalar("Loss/train", loss.item(), global_step)
               
                # Sauvegarde tous les 100 batches
                torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch+1}_step{idx}.pth")
                
        if (epoch + 1) % 10 == 0:
            embeddings = model.decoderRNN.embed.weight
            metadata = [full_dataset.vocab.idx2word[i] for i in range(len(full_dataset.vocab))]

            writer.add_embedding(
                embeddings,
                metadata=metadata,
                tag=f"word_embeddings_epoch_{epoch+1}"
            )

        torch.save(model.state_dict(), "model_latest.pth")

        bleu4 = evaluate_bleu(model, val_loader, full_dataset.vocab, device)  # <- Ici sur val_loader
        print(f"BLEU-4 score at epoch {epoch+1}: {bleu4:.4f}")
        writer.add_scalar("BLEU-4/val", bleu4, epoch+1)

    writer.close()

if __name__ == "__main__":
    train()
