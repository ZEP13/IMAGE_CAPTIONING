from PIL import Image #pillow permet de telcharger des images
import pandas as pd
import os 
import spacy 
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms # resize et normaliser les images
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset

spacy_eng = spacy.load("en_core_web_sm")

class Vocab:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def __len__(self):
        return len(self.idx2word)
    
    @staticmethod
    def tokenize(text):
        if not isinstance(text, str):
            return []
        return [token.text.lower().strip() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequence = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequence:
                    frequence[word] = 1
                else:
                    frequence[word] +=1
                
                if frequence[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx +=1

    def numericalize(self, text):
        tokenized_txt = self.tokenize(text)

        return [
            self.word2idx[token] if token in self.word2idx else self.word2idx["<UNK>"] 
            for token in tokenized_txt
        ]
            
            
class DatasetFLICK(Dataset):
    def __init__(self, dir, caption, transform, freq_threshold=5):
        self.dir = dir
        self.df = pd.read_csv(caption, sep='|')
        self.transform = transform

        # Nettoyage des colonnes
        self.df.columns = [col.strip() for col in self.df.columns]
        self.df["image_name"] = self.df["image_name"].astype(str).str.strip()
        self.df["comment"] = self.df["comment"].astype(str).str.strip()

        # Supprimer les lignes avec commentaires manquants
        self.df = self.df.dropna(subset=["comment"])

        self.vocab= Vocab(freq_threshold)
        self.vocab.build_vocab(self.df["comment"].tolist())

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        caption = self.df["comment"][index]
        imgid = self.df["image_name"][index]

        img = Image.open(os.path.join(self.dir, imgid)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.word2idx["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<EOS>"])

        return img, torch.tensor(numericalized_caption)



class MyCollate:
    """
        Quand tu charges des lots (batches) de données avec des séquences (captions), elles ont des longueurs différentes.
        Mais les tenseurs doivent avoir la même taille dans un batch → on doit les "padder" (ajout de 0).
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch): # batch est une liste de tuples : (image, caption)
        images = [item[0].unsqueeze(0) for item in batch]  # images
        captions = [item[1] for item in batch]             # captions (liste de tensors)

        images = torch.cat(images, dim=0)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions


def loader(
        root_folder,
        desc_file,
        transform,
        batchsize=16,
        worker=8,
        val_split=0.1,
        shuffle=True,
        memory=True
):
    full_dataset = DatasetFLICK(root_folder, desc_file, transform=transform)
    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])

    pad_idx = full_dataset.vocab.word2idx["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batchsize,
        num_workers=worker,
        shuffle=shuffle,
        pin_memory=memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batchsize,
        num_workers=worker,
        shuffle=False,
        pin_memory=memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return train_loader, val_loader, full_dataset



if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    train_loader, val_loader, dataset = loader(
        "flickr30k_images/flickr30k_images/", "flickr30k_images/results.csv", transform=transform
    )

    print("Train loader batches:")
    for idx, (imgs, captions) in enumerate(train_loader):
        print(f"Batch {idx}: imgs {imgs.shape}, captions {captions.shape}")
        if idx == 2:  # juste pour test, on stoppe au bout de 3 batches
            break

    print("\nValidation loader batches:")
    for idx, (imgs, captions) in enumerate(val_loader):
        print(f"Batch {idx}: imgs {imgs.shape}, captions {captions.shape}")
        if idx == 2:
            break


"""

Batch 0: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 49])
Batch 1: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 24])
Batch 2: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 29])

Validation loader batches:
Batch 0: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 33])
Batch 1: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 29])
Batch 2: imgs torch.Size([16, 3, 224, 224]), captions torch.Size([16, 52])


captions.shape = (16, X)
    16 = batch size

    X = longueur maximale de la séquence dans ce batch, dépend du padding
    → Elle varie selon les captions (par ex. 31, 25, 40...)

imgs.shape = (16, 3, 224, 224)
    16 = batch size

    3 = canaux RGB

    224x224 = image redimensionnée (comme demandé dans le transform)
"""