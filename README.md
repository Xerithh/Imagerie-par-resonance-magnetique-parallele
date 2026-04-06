Projet TP — pMRI (Imagerie par Résonance Magnétique parallèle)
=============================================================

But
---
Ce dépôt contient le code et les données pour un TP sur la reconstruction d'images en IRM parallèle (pMRI). Le but : simuler des acquisitions multi‑antenne sous différents niveaux de bruit et facteurs d'accélération `R`, implémenter une reconstruction SENSE (moindres carrés pondérés) et améliorer la solution avec une régularisation de Tikhonov.

Fichiers principaux
-------------------
- `main.py` : pipeline minimal qui exécute les 4 questions du TP (simulation R=2/R=4, reconstructions SENSE, balayage Tikhonov, génération de figures et impressions SNR).
- `utils.py` : fonctions utilitaires
  - `pMRI_simulator(S, ref, sigma, R)` : simule les mesures coil‑wise sous‑échantillonnées (aliasing).
  - `reconstruct(reduced_FoV, S, psi)` : implémentation SENSE / moindres carrés pondérés par `psi`.
  - `reconstruct_tikhonov(reduced_FoV, S, psi, lambd)` : reconstruction Tikhonov locale (régularisation quadratique).
  - `SignalToNoiseRatio(x_ref, x)` : calcule le SNR en dB.
- `reference.mat`, `sens.mat` : données fournies (image de référence et cartes de sensibilité).

Comment exécuter
----------------
1. Installer les dépendances (idéalement dans un environnement virtuel) :

```bash
pip install numpy scipy matplotlib
```

2. Lancer le script principal :

```bash
python main.py
```

Sorties générées
-----------------
Le script enregistre plusieurs fichiers images (PNG) dans le répertoire du projet :
- `q1_r2_noise.png` : simulations pour `R=2` (plusieurs sigma)
- `q2_r4_noise.png` : simulations pour `R=4` (plusieurs sigma)
- `q2_r2_vs_r4_sigma14.png` : comparaison R=2 vs R=4 pour `sigma=14` (figure demandée)
- `q3_sense_reconstruction.png` : reconstructions SENSE et cartes d'erreur
- `q4_lambda_curve.png` : courbe SNR(lambda) pour Tikhonov
- `q4_sense_vs_tikhonov.png` : comparaison SENSE vs meilleure Tikhonov

Paramètres modifiables
----------------------
- Dans `main.py` :
  - `sigma_values` : niveaux de bruit simulés.
  - `sigma_recon` : sigma utilisé pour construire `psi` et pour les reconstructions.
  - `lambda_values` : grille testée pour la régularisation Tikhonov.
  - `n_show` : nombre d'antennes affichées par figure (par défaut `min(4, Nc)`).

Observations (exemple de sortie)
--------------------------------
Lors d'une exécution, on observe typiquement :
- La qualité de reconstruction décroît quand `R` augmente (plus d'aliasing) ; SNR SENSE pour R=2 supérieure à R=4.
- La régularisation de Tikhonov (choix de `lambda`) peut fortement améliorer la SNR pour `R=4` ; la valeur optimale dépend du bruit et des cartes de sensibilité.

Rédaction / rapport
-------------------
- Question 4 demande une démonstration théorique que la solution du problème régularisé s'écrit
  $$(H^T \Psi^{-1} H + \lambda I)^{-1} H^T \Psi^{-1} z.$$ 
  Cette démonstration appartient au rapport écrit (algèbre des moindres carrés régularisés).

Notes
-----
- Le code est volontairement minimal et didactique pour suivre l'énoncé du TP. Si tu veux :
  - j'ajoute un `requirements.txt` ou un script d'installation, 
  - j'auto‑commit les modifications et crée un petit notebook pour visualiser interactif.

Contact
-------
Si tu veux que j'ajoute la démonstration, nettoie davantage le code, ou génère un README en anglais, dis‑le et je m'en charge.
