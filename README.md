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

Pour lancer un entraînement, utilisez la commande suivante depuis le dossier `test-NN/` :

```bash
python train.py [options]
```

### Options disponibles :

- `--run_name NAME` : Nom personnalisé pour cette exécution (par défaut : timestamp)
- `--split_method METHOD` : Méthode de division des données (choix : 'random' ou 'catchment')
  - `random` : Division aléatoire des données (par défaut)
  - `catchment` : Division par bassin versant
- `--test` : Active le mode test (divise les données en 70% train, 20% validation, 10% test)

### Exemples de commandes :

1. Entraînement standard avec division aléatoire :
```bash
python train.py
```

2. Entraînement avec un nom personnalisé :
```bash
python train.py --run_name "experiment_1"
```

3. Entraînement avec division par bassin versant :
```bash
python train.py --split_method "catchment"
```

4. Entraînement en mode test :
```bash
python train.py --test
```

5. Combinaison d'options :
```bash
python train.py --run_name "catchment_split_test" --split_method "catchment"
```

### Configuration avancée

Les paramètres du modèle peuvent être modifiés en définissant la variable d'environnement `MODEL_CONFIG` pointant vers un fichier JSON de configuration. Les paramètres par défaut sont :

```json
{
    "layer_dimensions": [4096],
    "dropout_rate": 0.1,
    "full_data": true,
    "num_blocks": 5,
    "layers_per_block": 4,
    "batch_size": 128,
    "years_range": [1994, 2022],
    "init_lr": 1e-15,
    "peak_lr": 1e-4,
    "final_lr": 1e-6,
    "warmup_epochs": 3,
    "constant_epochs": 5,
    "decay_epochs": 2
}
```

## Weights & Biases (wandb)

Weights & Biases est une plateforme gratuite de suivi d'expériences qui permet de visualiser et de comparer les performances de différents modèles. Dans ce projet, wandb est utilisé pour :

- Suivre les métriques d'entraînement (perte, taux d'apprentissage)
- Visualiser les prédictions du modèle
- Comparer les performances par bassin versant
- Sauvegarder les configurations d'entraînement
- Suivre l'évolution des métriques hydrologiques (KGE, R², etc.)

### Configuration de wandb

1. Créer un compte sur [Weights & Biases](https://wandb.ai)

2. Installer wandb :
```bash
pip install wandb
```

3. Se connecter à wandb (première utilisation) :
```bash
wandb login
```
Suivez les instructions pour obtenir et entrer votre clé API.

4. (Optionnel) Créer un nouveau projet sur wandb.ai et configurer le nom du projet :
```bash
export WANDB_PROJECT="baseflow-prediction"
```

### Utilisation dans le projet

Le projet utilise automatiquement wandb pour le suivi des expériences. Chaque exécution de `train.py` crée une nouvelle expérience dans wandb avec :

- Un nom unique basé sur la date et le nom personnalisé (si fourni)
- Les paramètres de configuration du modèle
- Les métriques d'entraînement en temps réel
- Les visualisations des prédictions
- Les performances par bassin versant

Les logs sont stockés localement dans le dossier `wandb/` et synchronisés avec le serveur wandb.

### Visualisation des résultats

1. Accédez à votre tableau de bord wandb : https://wandb.ai/[votre-username]/baseflow-prediction
2. Sélectionnez une exécution pour voir :
   - Les courbes d'apprentissage
   - Les visualisations des prédictions
   - Les métriques par bassin versant
   - Les configurations utilisées
