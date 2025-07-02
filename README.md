Transformation

🔍 Explication des points clés

✅ Grayscale (Niveaux de gris Lab a)

    Exploite le contraste vert / non-vert.

    PErmet ensuite de creer le masque.

✅ Masque (Mask)

    Créé à partir de l’image en niveaux de gris.

    Sert à isoler la plante, et enlever le fond.

    On crée le masque → puis on l’applique à l’image RGB.

✅ Gaussian Blur

    Floute légèrement l’image.

    Réduit le bruit et aide à la détection de contours ou à la robustesse du modèle.

    Pas obligatoire mais souvent utile.

✅ Landmarks

    Ce sont des points de repère sur la plante.

    Exemples : extrémités de feuilles, centre de la tige, etc.

    Utilisés en morphologie végétale pour détecter des formes, mesurer des angles, comparer des plantes.

    Avec plantcv.pseudolandmarks(), tu peux obtenir ces points automatiquement.

✅ Normalization

    Dernière étape avant d’envoyer au CNN.

    Pourquoi ?

        Les pixels vont de 0 à 255 → trop grande amplitude.

        CNN fonctionne mieux avec valeurs entre 0 et 1, ou centrées autour de 0.

                 +---------------------------+
                 |    Image RGB brute       |
                 |  (plante + fond)         |
                 +------------+-------------+
                              |
                              v
         +------------------------------------------+
         |   1. Convert to Grayscale (Lab - canal a)|   <- pour distinguer plante/fond
         +-------------------+----------------------+
                              |
                              v
         +------------------------------------------+
         |   2. Binary Mask (Thresholding)          |   <- créer masque binaire plante seule
         |   + Fill Holes / Clean mask              |
         +-------------------+----------------------+
                              |
                              v
         +------------------------------------------+
         |   3. Apply Mask                          |   <- applique le masque à l’image RGB
         |   => image = plante uniquement (fond blanc ou noir)  
         +-------------------+----------------------+
                              |
                              +---------------------------+
                              |                           |
                              v                           v
     +------------------------+--+        +------------------------------+
     |  4. Gaussian Blur (fac.)  |        |  5. Analyze Shape (Contours) |
     |  => lisser les bords      |        |  => surface, périmètre, etc.|
     +---------------------------+        +------------------------------+
                              |
                              v
     +----------------------------------------------------+
     |  6. ROI Detection (Contours / Bounding Box)        |
     |  => utile pour centrer / recadrer l’image          |
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
