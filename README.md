# Projet de Deep Learning

Ce projet contient des implémentations de réseaux de neurones pour l'analyse de données.

## Structure du Projet

```
DNN/
├── data/               # Dossier pour les données (non versionné)
├── test-NN/           # Tests unitaires
├── wandb/             # Logs de Weights & Biases (non versionné)
└── venv/              # Environnement virtuel Python (non versionné)
```

## Format des Données

Les données doivent être placées dans le dossier `data/` avec la structure suivante :

```
data/
├── train/            # Données d'entraînement
├── validation/       # Données de validation
└── test/            # Données de test
```

### Format des Fichiers
- Les données doivent être au format CSV
- Chaque fichier doit contenir les colonnes suivantes :
  - `features` : Les caractéristiques d'entrée
  - `target` : La variable cible

## Installation

1. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

[À compléter avec les instructions d'utilisation spécifiques à votre projet]

## Tests

Pour exécuter les tests :
```bash
python -m pytest test-NN/
``` 