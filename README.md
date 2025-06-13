# Projet de Deep Learning

Ce projet contient des implémentations de réseaux de neurones pour l'analyse de données.

## Structure du Projet

```
DNN/
├── data/               # Dossier pour les données (non versionné)
│   └── baseflow_23x100catchments_29years/  # Données de baseflow
├── test-NN/           # Code source et tests
├── wandb/             # Logs de Weights & Biases (non versionné)
└── venv/              # Environnement virtuel Python (non versionné)
```

## Format des Données

Les données doivent être placées dans le dossier `data/baseflow_23x100catchments_29years/` avec la structure suivante :

### Format des Fichiers
- Les données doivent être au format CSV
- Chaque fichier doit être nommé `baseflow_[catchment_name].csv` (ex: `baseflow_Allenbach_0.csv`)
- Chaque fichier doit contenir les colonnes suivantes :
  - `time` : Date au format datetime
  - `Q` : Débit total (streamflow) en m³/s
  - `baseflow` : Débit de base (baseflow) en m³/s
  - `Pmean` : Précipitations moyennes (optionnel, si full_data=True)
  - `Tmean` : Température moyenne (optionnel, si full_data=True)

### Prétraitement des Données
- Les données sont filtrées selon l'indice de baseflow (BFI) :
  - Seules les années avec 0.1 ≤ BFI ≤ 0.9 sont conservées
  - BFI = total_baseflow / total_streamflow
- Les données sont organisées en séquences annuelles (366 jours)
- Les données prétraitées sont mises en cache dans le dossier `preprocessed_data/`

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