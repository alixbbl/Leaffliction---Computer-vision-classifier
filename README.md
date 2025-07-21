# LEAFFLICTION

La premiere partie consiste a rendre 3 scripts : 
- Distribution.py : compte le nombre d'images dans un dossier, le but de ce script est de montrer que le dataset est inegal pour Apple et Grape.
- Augmentation.py : deux inputs possibles, une image ou un dossier. Ce script doit utiliser les transforms pour proposer un panel de data-augmentation possibles. La data-augmentation servant a pallier le manque d'images utilisables.
- Transformation.py : Utilisation de plantCV comme lib pour obtenir depuis les inages, la data exploitable pour le modele.

## Augmentation et Transformation

### 🔍 Explication des points clés

#### ✅ Grayscale (Niveaux de gris Lab a)
- Exploite le contraste vert / non-vert
- Permet ensuite de créer le masque

#### ✅ Masque (Mask)
- Créé à partir de l'image en niveaux de gris
- Sert à isoler la plante et enlever le fond
- On crée le masque → puis on l'applique à l'image RGB

#### ✅ Gaussian Blur
- Floute légèrement l'image
- Réduit le bruit et aide à la détection de contours ou à la robustesse du modèle
- Pas obligatoire mais souvent utile

#### ✅ Landmarks
- Ce sont des points de repère sur la plante
- **Exemples :** extrémités de feuilles, centre de la tige, etc.
- Utilisés en morphologie végétale pour détecter des formes, mesurer des angles, comparer des plantes
- Avec `plantcv.pseudolandmarks()`, on peut obtenir ces points automatiquement

#### ✅ Normalization
- Dernière étape avant d'envoyer au CNN
- **Pourquoi ?**
  - Les pixels vont de 0 à 255 → trop grande amplitude
  - CNN fonctionne mieux avec valeurs entre 0 et 1, ou centrées autour de 0

## Pipeline de Traitement

             +--------------------------+
             |    Image RGB brute       |
             |  (plante + fond)         |
             +------------+-------------+
                          |
                          v
     +------------------------------------------+
     |   1. Convert to Grayscale (Lab - canal a)|
     |   <- pour distinguer plante/fond         |
     +-------------------+----------------------+
                          |
                          v
     +------------------------------------------+
     |   2. Binary Mask (Thresholding)          |
     |   + Fill Holes / Clean mask              |
     |   <- créer masque binaire plante seule   |
     +-------------------+----------------------+
                          |
                          v
     +------------------------------------------+
     |   3. Apply Mask                          |
     |   <- applique le masque à l'image RGB    |
     |   => image = plante uniquement           |
     +-------------------+----------------------+
                          |
                          +---------------------------+
                          |                           |
                          v                           v
 +------------------------+--+        +------------------------------+
 |  4. Gaussian Blur (fac.)  |        |  5. Analyze Shape (Contours) |
 |  => lisser les bords      |        |  => surface, périmètre, etc. |
 +---------------------------+        +------------------------------+
                          |
                          v
 +----------------------------------------------------+
 |  6. ROI Detection (Contours / Bounding Box)        |
 |  => utile pour centrer / recadrer l'image          |
 +----------------------------------------------------+
                          |
                          v
 +----------------------------------------------------+
 |  7. Landmarks Detection (Pseudolandmarks) (option) |
 |  => repères botaniques, zones spécifiques          |
 +----------------------------------------------------+
                          |
                          v
 +----------------------------------------------------+
 |  8. Normalization                                   |
 |  => diviser par 255.0 ou normaliser par moyenne/std|
 |  => important pour entraîner un CNN                |
 +----------------------------------------------------+
                          |
                          v
 +------------------------------+
 |   Image prête pour le CNN    |
 +------------------------------+

 ## Train & Predict : Apprentissage et Prédictions

Le but du sujet est de coder un **CNN** (Convolutional Neural Networks), un modèle classique de Deep Learning couramment utilisé en reconnaissance d'images.

### Fonctionnement

Leur mode de fonctionnement est à première vue simple : l'utilisateur fournit en entrée une image sous la forme d'une matrice de pixels.

Celle-ci dispose de **3 dimensions** :
- **Deux dimensions** pour une image en niveaux de gris
- **Une troisième dimension** de profondeur 3 pour représenter les couleurs fondamentales (Rouge, Vert, Bleu)

### Architecture CNN

Contrairement à un modèle MLP (Multi Layers Perceptron) classique qui ne contient qu'une partie classification, l'architecture du Convolutional Neural Network dispose en amont d'une partie convolutive et comporte par conséquent **deux parties bien distinctes** :

#### 1. Partie Convolutive
Son objectif final est d'extraire des caractéristiques propres à chaque image en les compressant de façon à réduire leur taille initiale. 

**En résumé :** l'image fournie en entrée passe à travers une succession de filtres, créant par la même occasion de nouvelles images appelées **cartes de convolutions**. Enfin, les cartes de convolutions obtenues sont concaténées dans un vecteur de caractéristiques appelé **code CNN**.

#### 2. Partie Classification
Le code CNN obtenu en sortie de la partie convolutive est fourni en entrée dans une deuxième partie, constituée de couches entièrement connectées appelées **perceptron multicouche** (MLP pour Multi Layers Perceptron). 

**Le rôle de cette partie** est de combiner les caractéristiques du code CNN afin de classer l'image.