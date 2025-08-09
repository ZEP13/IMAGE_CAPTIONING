# Image Captioning - Génération Automatique de Descriptions d'Images

## Objectif

Ce projet vise à générer automatiquement des descriptions textuelles (captions) pour des images en utilisant des techniques de Deep Learning. Il s'appuie sur un modèle encodeur-décodeur (CNN + LSTM) entraîné sur le dataset Flickr30k.

## Architecture du Projet

- **loader.py** : Chargement, prétraitement des images et des légendes, gestion du vocabulaire.
- **model.py** : Définition du modèle `ImageCaptioningModel` (ResNet50 pré-entraîné comme encodeur + LSTM comme décodeur).
- **train.py** : Script d'entraînement du modèle, suivi des métriques, sauvegarde des checkpoints.
- **predict.py** : Génération de descriptions pour de nouvelles images à partir d'un modèle entraîné.
- **flickr30k_images/** : Dossier contenant les images et le fichier CSV des légendes.

## Installation

1. **Cloner le dépôt**

   ```bash
   git clone <url_du_repo>
   cd IMAGE_CAPTIONING
   ```

2. **Installer les dépendances**  
   Utilisez un environnement virtuel recommandé.

   ```bash
   pip install torch torchvision pillow pandas spacy tqdm tensorboard nltk
   python -m spacy download en_core_web_sm
   ```

3. **Télécharger le dataset Flickr30k**  
   Placez les images dans `flickr30k_images/flickr30k_images/` et le fichier `results.csv` dans `flickr30k_images/`.

## Utilisation

### Entraînement

Lancez l'entraînement du modèle :

```bash
python train.py
```

- Les checkpoints seront sauvegardés dans le dossier `checkpoints/`.
- Les logs TensorBoard sont dans `runs/image_captioning/`.

### Génération de description (inférence)

Pour générer une description sur une image :

```bash
python predict.py --image chemin/vers/image.jpg
```

La légende générée s'affichera dans le terminal.

## Structure du Modèle

- **Encodeur (CNN)** : ResNet50 pré-entraîné (ImageNet), dont la dernière couche est remplacée par une couche linéaire pour obtenir un vecteur d'embedding.
- **Décodeur (LSTM)** : Prend l'embedding de l'image et génère séquentiellement les mots de la légende.

## Personnalisation

- **Paramètres d'entraînement** : Modifiables dans `train.py` (taille d'embedding, learning rate, batch size, etc.).
- **Seuil de fréquence du vocabulaire** : Ajustable dans `loader.py` et les scripts principaux.

## Visualisation

Pour suivre l'entraînement et explorer les embeddings :

```bash
tensorboard --logdir runs/image_captioning
```

## Références

- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

Projet réalisé dans le cadre du cours de Machine Learning & Deep Learning.
