import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights #model pretrain ResNet
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Supprimer la dernière couche (fc) pour ne garder que les features
        modules = list(model.children())[:-1]  # enlève la couche fc

        self.resnet = nn.Sequential(*modules)

        # Ajouter une couche linéaire pour mapper les features sur embed_size
        self.linear = nn.Linear(model.fc.in_features, embed_size)

        # Dropout, pas besoin de relu deja dans le model
        self.dropout = nn.Dropout(0.3)

        for param in self.resnet.parameters():
            param.requires_grad = False  # on gèle tout d’abord

        for name, param in self.resnet.named_parameters():
            if "layer4" in name:
                param.requires_grad = True  # on dé-gèle layer4 uniquement
        
    def forward(self, images):
        """
            images: (batch_size, 3, 224, 224)
            return: (batch_size, embed_size)
        """

        features = self.resnet(images)                  # (batch_size, 2048, 1, 1) features est la sortie de ton resnet sans la dernière couche fc.
        features = features.view(features.size(0), -1) # (batch_size, 2048) changes la forme du tenseur sans toucher aux données
        features = self.linear(features)               # (batch_size, embed_size)
        features = self.dropout(features)              # régularisation
        return features                                # transformer la sortie 4D (batch, channels, height, width) en un vecteur 2D (batch, features)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        """
            vocab_size	Nombre total de mots dans ton vocab (taille du dictionnaire)

            embed_size	Dimension des vecteurs de mots, par ex. 256 ou 512

            hidden_size	Taille de l’état caché (par exemple 512 ou 1024)

            num_layers	Nombre de couches LSTM empilées (souvent 1 ou 2)
        """
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.5) # Probabilité de désactiver un neurone pendant l'entraînement
    
    def forward(self, features, captions, lengths=None):
        """
        features: (batch_size, embed_size)
        captions: (batch_size, max_len)
        lengths: longueurs réelles des captions (optionnel mais recommandé pour pack_padded)



        Le LSTM ignore les <PAD> pendant l'apprentissage (meilleure efficacité)
        """
        # 1. Embedding
        embeddings = self.embed(captions[:, :-1])  # (batch, seq_len-1, embed_size)

        # 2. Ajouter le vecteur d’image au début de chaque séquence
        features = features.unsqueeze(1)  # (batch, 1, embed_size)
        embeddings = torch.cat((features, embeddings), dim=1)  # (batch, seq_len, embed_size)

        # 3. Mettre à jour les longueurs (chaque séquence a +1 à cause de l’image)
        if lengths is not None:
            lengths = [l + 1 for l in lengths]  # +1 pour le token image
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeddings)

        lstm_out = self.drop(lstm_out)
        outputs = self.linear(lstm_out)  # (batch, seq_len, vocab_size)
        return outputs
        

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
        
        
    def forward(self, images, captions, lengths=None):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, lengths)
        return outputs
    
    def caption(self, image, vocab, max_length=50):
        result = []

        device = next(self.encoderCNN.parameters()).device
        image = image.to(device)
        # Encode image
        with torch.no_grad():
            # image doit être de forme (1, 3, H, W)
            if image.dim() == 3:
                feature = self.encoderCNN(image.unsqueeze(0))  # (1, embed_size)
            else:
                feature = self.encoderCNN(image)  # (1, embed_size)

        # Initial LSTM input = feature, no hidden state
        inputs = feature.unsqueeze(1)  # (1, 1, embed_size)
        states = None

        for _ in range(max_length):
            hiddens, states = self.decoderRNN.lstm(inputs, states)  # (1, 1, hidden_size)
            output = self.decoderRNN.linear(hiddens.squeeze(1))     # (1, vocab_size)
            predicted = output.argmax(1)                            # (1,) index du mot
            word = vocab.idx2word[predicted.item()]
            result.append(word)

            if word.lower() == "<eos>":
                break

            # 3. Reinject predicted word as input
            inputs = self.decoderRNN.embed(predicted).unsqueeze(1)  # (1, 1, embed_size)

        return result