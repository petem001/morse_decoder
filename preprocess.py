# =============================================================================
# SCRIPT DE PRÉTRAITEMENT DES DONNÉES AUDIO MORSE
# =============================================================================
# Ce programme transforme les fichiers audio WAV bruts en données utilisables
# par le modèle d'IA. Il extrait les caractéristiques MFCC qui permettent
# au modèle de "comprendre" les signaux audio.
# =============================================================================

# IMPORTATION DES BIBLIOTHÈQUES
# ------------------------------
import os           # Pour naviguer dans les dossiers et fichiers
import json         # Pour créer et lire des fichiers de configuration
import numpy as np  # Pour les calculs mathématiques sur les tableaux
import librosa      # Bibliothèque spécialisée dans l'analyse audio
from tqdm import tqdm              # Pour afficher des barres de progression
from collections import Counter   # Pour compter les caractères

# CONFIGURATION DU PRÉTRAITEMENT
# -------------------------------
# Ces paramètres définissent comment l'audio sera analysé
SR = 22050              # Fréquence d'échantillonnage (Hz) - réduite pour l'efficacité
HOP_LENGTH = 512        # Taille du saut entre les fenêtres d'analyse
N_MFCC = 13            # Nombre de coefficients MFCC à extraire
MAX_DURATION = 60.0     # Durée maximale d'un fichier audio (secondes)
OUTPUT_DIR = "preprocessed_data"  # Dossier de sortie pour les données traitées

def extract_features(audio_path):
    """
    EXTRACTION DES CARACTÉRISTIQUES AUDIO (MFCC)
    =============================================
    Cette fonction est le cœur du prétraitement. Elle transforme un fichier
    audio WAV en caractéristiques MFCC que le modèle d'IA peut comprendre.
    
    Les MFCC (Mel-Frequency Cepstral Coefficients) représentent les 
    caractéristiques spectrales importantes de l'audio, similaires à la 
    façon dont l'oreille humaine perçoit le son.
    
    Arguments:
    - audio_path: Chemin vers le fichier audio à analyser
    
    Retourne:
    - Un tableau numpy de forme (temps, 39) contenant les caractéristiques
    - None si une erreur survient
    """
    try:
        # CHARGEMENT DU FICHIER AUDIO
        # ---------------------------
        # librosa.load() lit le fichier et le convertit au bon format
        audio, _ = librosa.load(audio_path, sr=SR, duration=MAX_DURATION)
        
        # EXTRACTION DES MFCC DE BASE (13 COEFFICIENTS)
        # ---------------------------------------------
        # Les MFCC capturent les caractéristiques spectrales du signal
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        
        # CALCUL DES DÉRIVÉES (DELTA ET DELTA-DELTA)
        # -------------------------------------------
        # Les dérivées première et seconde ajoutent des informations
        # sur l'évolution temporelle des caractéristiques
        delta = librosa.feature.delta(mfcc)          # Dérivée première (vitesse de changement)
        delta2 = librosa.feature.delta(mfcc, order=2) # Dérivée seconde (accélération)
        
        # CONCATÉNATION DES CARACTÉRISTIQUES
        # -----------------------------------
        # On combine MFCC + Delta + Delta2 = 13 + 13 + 13 = 39 caractéristiques
        # .T transpose la matrice pour avoir la forme (temps, caractéristiques)
        return np.concatenate([mfcc, delta, delta2], axis=0).T
        
    except Exception as e:
        print(f"Erreur lors du traitement de {audio_path}: {str(e)}")
        return None

def preprocess_dataset(input_dir):
    """
    PRÉTRAITEMENT COMPLET DU DATASET
    ================================
    Cette fonction traite tous les fichiers audio d'un dossier et prépare
    les données pour l'entraînement du modèle d'IA.
    
    Processus:
    1. Parcourt tous les fichiers .wav du dossier d'entrée
    2. Pour chaque fichier audio, cherche le fichier .txt correspondant
    3. Extrait les caractéristiques MFCC de l'audio
    4. Valide que les données sont exploitables
    5. Sauvegarde tout dans un format optimisé
    6. Crée un vocabulaire de tous les caractères utilisés
    
    Arguments:
    - input_dir: Dossier contenant les fichiers audio (.wav) et texte (.txt)
    """
    
    # CRÉATION DES DOSSIERS DE SORTIE
    # --------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)           # Dossier principal
    mfcc_dir = os.path.join(OUTPUT_DIR, "mfccs")     # Sous-dossier pour les MFCC
    os.makedirs(mfcc_dir, exist_ok=True)

    # INITIALISATION DES STRUCTURES DE DONNÉES
    # -----------------------------------------
    metadata = []           # Liste des informations sur chaque fichier
    char_counter = Counter() # Compteur pour créer le vocabulaire
    
    # Trouver tous les fichiers audio
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # TRAITEMENT DE CHAQUE FICHIER AUDIO
    # ----------------------------------
    print(f"Traitement de {len(audio_files)} fichiers audio...")
    
    for audio_file in tqdm(audio_files, desc="Extraction des caractéristiques"):
        # RECHERCHE DU FICHIER TEXTE CORRESPONDANT
        # -----------------------------------------
        base_name = os.path.splitext(audio_file)[0]  # Nom sans extension
        text_file = os.path.join(input_dir, f"{base_name}.txt")

        # Ignorer si pas de transcription correspondante
        if not os.path.exists(text_file):
            print(f"⚠️  Fichier texte manquant pour {audio_file}")
            continue

        # LECTURE DE LA TRANSCRIPTION
        # ---------------------------
        with open(text_file, 'r', encoding='utf-8') as f:
            text_label = f.read().strip().upper()  # Lecture + nettoyage

        # VALIDATION DU CONTENU TEXTUEL
        # -----------------------------
        # Vérifier que le texte a une longueur raisonnable
        if not 3 <= len(text_label) <= 100:
            print(f"⚠️  Texte trop court/long pour {audio_file}: {len(text_label)} caractères")
            continue

        # EXTRACTION DES CARACTÉRISTIQUES AUDIO
        # --------------------------------------
        mfcc = extract_features(os.path.join(input_dir, audio_file))
        
        # Ignorer si l'extraction a échoué
        if mfcc is None:
            print(f"⚠️  Échec extraction pour {audio_file}")
            continue

        # SAUVEGARDE DES DONNÉES TRAITÉES
        # --------------------------------
        # Sauvegarder les MFCC dans un fichier numpy (.npy)
        np.save(os.path.join(mfcc_dir, f"{base_name}.npy"), mfcc)
        
        # Compter les caractères pour le vocabulaire
        char_counter.update(text_label)
        
        # Ajouter aux métadonnées
        metadata.append({
            "audio_file": audio_file,              # Nom du fichier audio original
            "mfcc_file": f"{base_name}.npy",      # Nom du fichier MFCC
            "text_label": text_label,             # Transcription textuelle
            "duration": mfcc.shape[0] * HOP_LENGTH / SR  # Durée calculée
        })

    # CRÉATION DU VOCABULAIRE
    # -----------------------
    # Créer un dictionnaire qui associe chaque caractère à un numéro
    # (nécessaire car les modèles d'IA ne comprennent que les nombres)
    char_to_int = {char: i+1 for i, char in enumerate(sorted(char_counter.keys()))}
    # Note: l'indice 0 sera réservé pour le padding (remplissage)

    # SAUVEGARDE DES MÉTADONNÉES ET CONFIGURATION
    # --------------------------------------------
    metadata_content = {
        "config": {
            "sample_rate": SR,          # Paramètres utilisés
            "hop_length": HOP_LENGTH,
            "n_mfcc": N_MFCC
        },
        "files": metadata,              # Informations sur tous les fichiers
        "char_map": char_to_int        # Dictionnaire caractère -> nombre
    }
    
    # Écriture du fichier de métadonnées
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata_content, f, indent=2, ensure_ascii=False)

    # RÉSUMÉ DU TRAITEMENT
    # --------------------
    print(f"\n✅ Prétraitement terminé avec succès!")
    print(f"📊 Statistiques:")
    print(f"   - Fichiers traités: {len(metadata)}")
    print(f"   - Caractères uniques: {len(char_counter)}")
    print(f"   - Vocabulaire: {sorted(char_counter.keys())}")
    print(f"📁 Données sauvegardées dans: {OUTPUT_DIR}/")
    print(f"   - MFCC: {OUTPUT_DIR}/mfccs/")
    print(f"   - Métadonnées: {OUTPUT_DIR}/metadata.json")

# =============================================================================
# EXÉCUTION DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Lancer le prétraitement sur le dossier "mon_dataset"
    # Vous pouvez changer ce nom selon votre configuration
    input_directory = "mon_dataset"
    
    # Vérifier que le dossier existe
    if not os.path.exists(input_directory):
        print(f"❌ Erreur: Le dossier '{input_directory}' n'existe pas!")
        print(f"   Assurez-vous d'avoir généré votre dataset avec le générateur Morse.")
        print(f"   Ou changez 'input_directory' dans ce script.")
    else:
        print(f"🚀 Début du prétraitement du dossier: {input_directory}")
        preprocess_dataset(input_directory)

# =============================================================================
# INFORMATIONS IMPORTANTES POUR L'UTILISATEUR
# =============================================================================
"""
COMMENT UTILISER CE SCRIPT:
===========================

1. PRÉREQUIS:
   - Avoir un dossier contenant des paires de fichiers .wav et .txt
   - Chaque fichier audio doit avoir sa transcription correspondante
   - Exemple: morse_0001.wav + morse_0001.txt

2. INSTALLATION DES DÉPENDANCES:
   pip install librosa numpy tqdm

3. EXÉCUTION:
   python preprocess.py

4. RÉSULTATS:
   - Dossier "preprocessed_data/" créé
   - Fichiers MFCC dans "preprocessed_data/mfccs/"
   - Fichier "metadata.json" avec toutes les informations

5. PERSONNALISATION:
   - Modifier 'input_directory' pour changer le dossier source
   - Ajuster SR, N_MFCC, etc. selon vos besoins
   - MAX_DURATION limite la longueur des fichiers audio

DÉPANNAGE:
==========
- "Fichier texte manquant": Vérifiez que chaque .wav a son .txt
- "Texte trop court/long": Ajustez les limites dans la validation
- "Échec extraction": Vérifiez que les fichiers audio sont valides
- Erreurs de mémoire: Réduisez MAX_DURATION ou N_MFCC
"""