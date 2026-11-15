
"""
download_datasets.py
---------------------
Telecharge les 2 datasets necessaires pour le projet :

1. atlasia/DODa-audio-dataset
   Dataset utilise pour la reconnaissance automatique de la parole (ASR).
   Il contient des fichiers audio et leurs transcriptions en Darija
   (alphabet latin et arabe) et en anglais.

2. MBZUAI-Paris/Darija-SFT-Mixture
   Dataset utilise pour la traduction.
   Il contient plusieurs types de donnees linguistiques.
   Nous allons ensuite filtrer les paires Darija -> Francais.

Usage:
    python scripts/download_datasets.py

Necessite une connexion internet vers huggingface.co
(a lancer sur Colab ou sur ta machine perso, pas dans un environnement reseau restreint).
"""

# Importation de la bibliotheque os pour manipuler les chemins et dossiers.
import os

# Importation de load_dataset pour telecharger les datasets depuis Hugging Face.
from datasets import load_dataset


# Creation du chemin vers le dossier qui va contenir les donnees brutes.
# os.path.dirname(__file__) permet de recuperer le dossier du script actuel.
# Les donnees seront stockees dans : data/raw/
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Creation du dossier data/raw s'il n'existe pas deja.
# exist_ok=True evite une erreur si le dossier existe deja.
os.makedirs(RAW_DATA_DIR, exist_ok=True)


# Fonction responsable du telechargement du dataset audio pour l'ASR.
def download_asr_dataset():
    """Telecharge le dataset audio darija (DODa-audio-dataset)."""

    # Affichage d'un message pour informer l'utilisateur du debut du telechargement.
    print("Telechargement de atlasia/DODa-audio-dataset ...")

    # Telechargement du dataset depuis la plateforme Hugging Face.
    ds = load_dataset("atlasia/DODa-audio-dataset")

    # Definition du chemin de sauvegarde du dataset audio.
    save_path = os.path.join(RAW_DATA_DIR, "doda_audio_dataset")

    # Sauvegarde du dataset telecharge sur le disque.
    ds.save_to_disk(save_path)

    # Affichage du chemin ou les donnees ont ete sauvegardees.
    print(f"  -> Sauvegarde dans {save_path}")

    # Affichage des differentes parties (train, validation, test...) disponibles.
    print(f"  -> Splits disponibles: {list(ds.keys())}")

    # Parcours de chaque partie du dataset pour afficher son nombre d'exemples.
    for split in ds.keys():
        print(f"     {split}: {len(ds[split])} exemples")

    # Retourne le dataset pour pouvoir l'utiliser dans la suite du projet.
    return ds


# Fonction responsable du telechargement du dataset de traduction.
def download_translation_dataset():
    """Telecharge le dataset de traduction (Darija-SFT-Mixture) et filtre darija->francais."""

    # Affichage d'un message indiquant le debut du telechargement.
    print("\nTelechargement de MBZUAI-Paris/Darija-SFT-Mixture ...")

    # Telechargement du dataset depuis Hugging Face.
    ds = load_dataset("MBZUAI-Paris/Darija-SFT-Mixture")

    # Definition du chemin de sauvegarde du dataset complet.
    save_path = os.path.join(RAW_DATA_DIR, "darija_sft_mixture_full")

    # Sauvegarde de toutes les donnees telechargees.
    ds.save_to_disk(save_path)

    # Affichage du chemin de sauvegarde.
    print(f"  -> Sauvegarde complete dans {save_path}")

    # Affichage des parties disponibles dans le dataset.
    print(f"  -> Splits disponibles: {list(ds.keys())}")

    # Affichage du nombre d'exemples pour chaque partie du dataset.
    for split in ds.keys():
        print(f"     {split}: {len(ds[split])} exemples")

    # Retourne le dataset pour les prochaines etapes du projet.
    return ds


# Ce bloc est execute uniquement lorsque le fichier est lance directement.
# Il ne sera pas execute si ce fichier est importe dans un autre script.
if __name__ == "__main__":

    # Telechargement du dataset utilise pour la reconnaissance vocale (ASR).
    asr_ds = download_asr_dataset()

    # Telechargement du dataset utilise pour la traduction.
    translation_ds = download_translation_dataset()

    # Message indiquant que les deux telechargements sont termines.
    print("\nTelechargement termine. Donnees brutes dans data/raw/")

    # Indication du prochain notebook a executer : exploration des donnees.
    print("Prochaine etape: notebooks/data_exploration.ipynb")
