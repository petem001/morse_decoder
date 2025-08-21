# =============================================================================
# GÉNÉRATEUR DE DATASET MORSE AUTOMATISÉ
# =============================================================================
# Ce programme génère automatiquement des milliers de fichiers audio contenant
# du code Morse et leurs transcriptions textuelles correspondantes.
# Il simule différentes conditions réalistes (bruit, variations humaines, etc.)
# =============================================================================

# IMPORTATION DES BIBLIOTHÈQUES
# ------------------------------
import json                    # Pour lire les fichiers de configuration
import numpy as np            # Pour les calculs mathématiques
import scipy.io.wavfile as wavfile  # Pour sauvegarder les fichiers audio
import os                     # Pour gérer les fichiers et dossiers
import sys                    # Pour les arguments de ligne de commande
import random                 # Pour la génération aléatoire
import matplotlib.pyplot as plt  # Pour les graphiques (si nécessaire)
from tqdm import tqdm        # Pour les barres de progression
from scipy import signal    # Pour le traitement du signal
from scipy.io import savemat # Pour sauvegarder en format MATLAB
import glob                  # Pour rechercher des fichiers
import re                    # Pour les expressions régulières

# CONSTANTES AUDIO DE BASE
# ------------------------
SAMPLE_RATE = 44100      # Fréquence d'échantillonnage standard (CD quality)
MORSE_TONE_FREQ = 700    # Fréquence du signal Morse en Hz (ton medium)
MAX_AMPLITUDE = 0.5      # Amplitude maximale pour éviter la saturation

# FRÉQUENCES NATURELLES DES LETTRES EN FRANÇAIS/ANGLAIS
# ------------------------------------------------------
# Ces pourcentages reflètent la fréquence d'apparition naturelle des lettres
# dans les textes courants. Permet de générer du texte plus réaliste.
NATURAL_FREQ = {
    'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 
    'N': 6.7, 'S': 6.3, 'H': 6.1, 'R': 6.0, 'D': 4.3,
    'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4, 'W': 2.4,
    'F': 2.2, 'G': 2.0, 'Y': 2.0, 'P': 1.9, 'B': 1.5,
    'V': 1.0, 'K': 0.8, 'J': 0.2, 'X': 0.2, 'Q': 0.1,
    'Z': 0.1
}

def find_next_file_number(output_dir):
    """
    RECHERCHE DU PROCHAIN NUMÉRO DE FICHIER DISPONIBLE
    ==================================================
    Cette fonction permet de continuer la numérotation sans écraser
    les fichiers existants lors de générations multiples.
    
    Arguments:
    - output_dir: Dossier où chercher les fichiers existants
    
    Retourne:
    - Le prochain numéro disponible (entier)
    """
    if not os.path.exists(output_dir):
        return 0  # Premier fichier sera morse_0000
    
    # Chercher tous les fichiers morse_*.txt existants
    existing_files = glob.glob(os.path.join(output_dir, 'morse_*.txt'))
    numbers = []
    
    # Extraire les numéros des noms de fichiers
    for f in existing_files:
        match = re.search(r'morse_(\d+)\.txt', f)
        if match:
            numbers.append(int(match.group(1)))
    
    # Retourner le numéro suivant le plus grand trouvé
    return max(numbers) + 1 if numbers else 0

def generate_tone(frequency, duration, sample_rate, amplitude, freq_drift_enabled=False, max_variation_hz=0):
    """
    GÉNÉRATION D'UN SIGNAL SINUSOÏDAL (TONALITÉ MORSE)
    ==================================================
    Cette fonction crée le son de base du code Morse : un signal sinusoïdal pur.
    Elle peut aussi simuler la dérive de fréquence des anciens équipements radio.
    
    Arguments:
    - frequency: Fréquence du signal en Hz
    - duration: Durée du signal en secondes
    - sample_rate: Fréquence d'échantillonnage
    - amplitude: Amplitude du signal (0-1)
    - freq_drift_enabled: Activer la dérive de fréquence
    - max_variation_hz: Variation maximale de fréquence
    
    Retourne:
    - Tableau numpy contenant le signal audio
    """
    # Créer l'axe temporel
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if freq_drift_enabled and max_variation_hz > 0:
        # SIMULATION DE LA DÉRIVE DE FRÉQUENCE
        # ------------------------------------
        # Les anciens équipements radio avaient des oscillateurs instables
        # qui causaient une légère variation de fréquence au fil du temps
        drift = max_variation_hz * np.sin(2 * np.pi * 0.5 * t / duration)
        current_frequency = frequency + drift
        audio_data = amplitude * np.sin(2 * np.pi * current_frequency * t)
    else:
        # Signal sinusoïdal pur standard
        audio_data = amplitude * np.sin(2 * np.pi * frequency * t)
        
    return audio_data

def add_noise(audio_data, noise_type, snr_db, sample_rate):
    """
    AJOUT DE BRUIT DE FOND RÉALISTE
    ===============================
    Cette fonction simule les conditions réelles d'écoute radio en ajoutant
    différents types de bruit de fond au signal Morse.
    
    Types de bruit simulés:
    - Bruit blanc: Toutes les fréquences également présentes
    - Bruit rose: Plus de basses fréquences (plus naturel)
    - Bruit brownien: Encore plus de basses fréquences
    
    Arguments:
    - audio_data: Signal audio original
    - noise_type: Type de bruit ("white", "pink", "brownian", "none")
    - snr_db: Rapport signal/bruit en décibels (plus élevé = moins de bruit)
    - sample_rate: Fréquence d'échantillonnage
    
    Retourne:
    - Signal audio avec bruit ajouté
    """
    if noise_type == "none":
        return audio_data

    # Calculer la puissance du signal original
    signal_power = np.mean(audio_data**2)
    
    if signal_power == 0:  # Éviter la division par zéro
        return audio_data

    # Convertir SNR de dB en rapport linéaire
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.zeros_like(audio_data)

    if noise_type == "white":
        # BRUIT BLANC: toutes les fréquences avec la même intensité
        noise = np.random.normal(0, 1, len(audio_data))
        
    elif noise_type == "pink":
        # BRUIT ROSE: plus d'énergie dans les basses fréquences
        # Simule mieux le bruit atmosphérique naturel
        unfiltered_noise = np.random.normal(0, 1, len(audio_data))
        # Filtre passe-bas simple pour créer l'effet "rose"
        b, a = signal.butter(1, 0.05, btype='low', analog=False, fs=sample_rate)
        noise = signal.lfilter(b, a, unfiltered_noise)
        
    elif noise_type == "brownian":
        # BRUIT BROWNIEN: encore plus de basses fréquences
        # Simulation du bruit de fond électronique
        unfiltered_noise = np.random.normal(0, 1, len(audio_data))
        noise = np.cumsum(unfiltered_noise)  # Intégration = filtre passe-bas
    
    # Normaliser le bruit à la puissance désirée
    if np.mean(noise**2) > 0:  # Éviter la division par zéro
        noise = noise * np.sqrt(noise_power / (np.mean(noise**2) + 1e-9))
    
    return audio_data + noise

def apply_fading(audio_data, max_depth_db, speed_variation, sample_rate):
    """
    APPLICATION D'UN EFFET DE FADING (ÉVANOUISSEMENT)
    =================================================
    Le fading simule les variations d'intensité du signal causées par
    la propagation radio (réflexions ionosphériques, etc.)
    
    Arguments:
    - audio_data: Signal audio
    - max_depth_db: Profondeur maximale du fading en dB
    - speed_variation: Vitesse de variation du fading
    - sample_rate: Fréquence d'échantillonnage
    
    Retourne:
    - Signal audio avec effet de fading appliqué
    """
    if max_depth_db <= 0:
        return audio_data

    duration = len(audio_data) / sample_rate
    t = np.linspace(0, duration, len(audio_data), endpoint=False)

    # Convertir la profondeur dB en facteur d'amplitude linéaire
    min_amplitude_factor = 10**(-max_depth_db / 20)
    max_amplitude_factor = 1.0

    # Créer une enveloppe sinusoïdale de modulation
    fading_envelope = ((1 - min_amplitude_factor) / 2) * \
                     (np.cos(2 * np.pi * speed_variation * t) + 1) + min_amplitude_factor
    
    return audio_data * fading_envelope

def apply_distortion(audio_data, dist_type):
    """
    APPLICATION D'EFFETS DE DISTORSION
    ==================================
    Simule la distorsion causée par des équipements radio de mauvaise qualité
    ou la surcharge des amplificateurs.
    
    Arguments:
    - audio_data: Signal audio
    - dist_type: Type de distorsion ("soft", "hard", "none")
    
    Retourne:
    - Signal audio avec distorsion appliquée
    """
    if dist_type == "none":
        return audio_data
    
    # Normaliser temporairement pour éviter l'écrêtage excessif
    max_abs_val = np.max(np.abs(audio_data))
    if max_abs_val > 0:
        normalized_audio = audio_data / max_abs_val
    else:
        return audio_data

    if dist_type == "soft":
        # DISTORSION DOUCE: utilise la fonction tanh
        # Simule la saturation progressive des tubes électroniques
        distorted_audio = np.tanh(normalized_audio * 2.0)
        
    elif dist_type == "hard":
        # DISTORSION DURE: écrêtage brutal
        # Simule la surcharge des circuits numériques
        threshold = 0.5
        distorted_audio = np.clip(normalized_audio, -threshold, threshold) / threshold
        
    else:
        return audio_data

    # Restaurer l'amplitude originale
    return distorted_audio * max_abs_val

def add_impulse_noise(audio_data, probability, magnitude):
    """
    AJOUT DE BRUIT IMPULSIONNEL
    ===========================
    Simule les "craquements" et interférences ponctuelles typiques
    des communications radio (parasites électriques, orages, etc.)
    
    Arguments:
    - audio_data: Signal audio
    - probability: Probabilité d'occurrence des impulsions
    - magnitude: Intensité des impulsions
    
    Retourne:
    - Signal audio avec impulsions ajoutées
    """
    if probability <= 0:
        return audio_data
    
    num_samples = len(audio_data)
    num_impulses = int(num_samples * probability)

    # Sélectionner des positions aléatoires
    impulse_indices = np.random.choice(num_samples, num_impulses, replace=False)

    # Générer des impulsions aléatoires (positives ou négatives)
    impulse_values = (np.random.rand(num_impulses) * 2 - 1) * magnitude
    
    # Appliquer les impulsions (modification directe du signal)
    audio_data[impulse_indices] += impulse_values
    
    return audio_data

def get_morse_duration(wpm, char_type="dot"):
    """
    CALCUL DES DURÉES MORSE STANDARD
    ================================
    Le code Morse a des durées normalisées basées sur la vitesse en WPM.
    Cette fonction calcule les durées correctes pour chaque élément.
    
    Standard international:
    - 1 "dit" (point) = unité de base
    - 1 "dah" (trait) = 3 dits
    - Espace intra-caractère = 1 dit
    - Espace inter-caractère = 3 dits  
    - Espace inter-mot = 7 dits
    
    Arguments:
    - wpm: Vitesse en mots par minute
    - char_type: Type d'élément ("dot", "dash", "intra_char_space", etc.)
    
    Retourne:
    - Durée en secondes
    """
    # Formule standard: 1200ms par dit à 1 WPM
    dit_duration_ms = 1200 / wpm

    if char_type == "dot":
        return dit_duration_ms / 1000  # Point = 1 dit
    elif char_type == "dash":
        return (dit_duration_ms * 3) / 1000  # Trait = 3 dits
    elif char_type == "intra_char_space":  
        return dit_duration_ms / 1000  # Espace dans le caractère = 1 dit
    elif char_type == "char_space":  
        return (dit_duration_ms * 3) / 1000  # Espace entre caractères = 3 dits
    elif char_type == "word_space":  
        return (dit_duration_ms * 7) / 1000  # Espace entre mots = 7 dits
    else:
        return 0

def apply_human_variability(duration, min_var, max_var):
    """
    APPLICATION DE LA VARIABILITÉ HUMAINE
    =====================================
    Les opérateurs humains n'envoient jamais le Morse avec une précision parfaite.
    Cette fonction simule les variations naturelles de timing.
    
    Arguments:
    - duration: Durée de base
    - min_var: Variation minimale en pourcentage (négative)
    - max_var: Variation maximale en pourcentage (positive)
    
    Retourne:
    - Durée modifiée avec variabilité humaine
    """
    variation_percent = random.uniform(min_var, max_var)
    return duration * (1 + variation_percent / 100)

def generate_random_text(num_chars, char_types, char_frequencies, morse_dicts):
    """
    GÉNÉRATION DE TEXTE ALÉATOIRE INTELLIGENT
    =========================================
    Cette fonction crée du texte aléatoire en respectant les fréquences
    naturelles des lettres ou des pondérations personnalisées.
    
    Arguments:
    - num_chars: Nombre de caractères à générer
    - char_types: Types de caractères à inclure (alphabet, chiffres, etc.)
    - char_frequencies: Configuration des fréquences
    - morse_dicts: Dictionnaires de code Morse
    
    Retourne:
    - Chaîne de texte générée aléatoirement
    """
    all_possible_chars = []

    # AJOUT DES CARACTÈRES ALPHABÉTIQUES
    # -----------------------------------
    if "alphabet" in char_types:
        alphabet_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        if char_frequencies['alphabet'] == 'natural':
            # FRÉQUENCES NATURELLES: utilise les statistiques réelles
            weights = [NATURAL_FREQ.get(char, 0.1) for char in alphabet_chars]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            # Créer une grande réserve avec les bonnes proportions
            all_possible_chars.extend(
                random.choices(alphabet_chars, weights=probabilities, k=5000)
            )
        else:  # Distribution uniforme
            all_possible_chars.extend(alphabet_chars * 200)
    
    # AJOUT DES CHIFFRES
    if "numbers" in char_types:
        all_possible_chars.extend(list("0123456789") * 100)
    
    # AJOUT DE LA PONCTUATION
    if "punctuation" in char_types:
        punctuation_chars = [
            char for char in morse_dicts["MORSE_CODE_DICT"].keys() 
            if char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ]
        all_possible_chars.extend(punctuation_chars * 20)

    # AJOUT DES CARACTÈRES ACCENTUÉS
    if "accents" in char_types:
        all_possible_chars.extend(
            list(morse_dicts["MORSE_ACCENTED_CHARS_DICT"].keys()) * 5
        )
    
    # AJOUT DES PROSIGNS (SIGNAUX DE PROCÉDURE)
    if "prosigns" in char_types:
        all_possible_chars.extend(list(morse_dicts["PROSIGN_DICT"].keys()) * 5)

    # GESTION DES FRÉQUENCES PERSONNALISÉES
    if char_frequencies['alphabet'] == 'custom':
        custom_chars = list(char_frequencies['customWeights'].keys())
        custom_weights = list(char_frequencies['customWeights'].values())
        total_custom_weight = sum(custom_weights)
        if total_custom_weight > 0:
            custom_probabilities = [w / total_custom_weight for w in custom_weights]
            all_possible_chars.extend(
                random.choices(custom_chars, weights=custom_probabilities, k=5000)
            )

    if not all_possible_chars:
        return ""  # Aucun type de caractère sélectionné

    # GÉNÉRATION DU TEXTE FINAL
    generated_text = []
    for _ in range(num_chars):
        char = random.choice(all_possible_chars)
        
        # Gestion spéciale des prosigns multi-caractères
        if char in morse_dicts["PROSIGN_DICT"] and "prosigns" in char_types:
            if random.random() < 0.2:  # 20% de chance d'utiliser un prosign
                generated_text.append(char)
            else:  # Sinon, prendre un caractère simple
                single_chars = [c for c in all_possible_chars if len(c) == 1]
                if single_chars:
                    generated_text.append(random.choice(single_chars))
        else:
            generated_text.append(char)
            
    return "".join(generated_text)

def add_special_sequences(text, special_sequences_config, morse_dicts):
    """
    AJOUT DE SÉQUENCES SPÉCIALES (CODES Q, ABRÉVIATIONS)
    ====================================================
    Les communications radio utilisent des codes standardisés et des
    abréviations pour la concision et la clarté.
    
    Arguments:
    - text: Texte de base
    - special_sequences_config: Configuration des séquences spéciales  
    - morse_dicts: Dictionnaires contenant les codes
    
    Retourne:
    - Texte enrichi avec les séquences spéciales
    """
    if not special_sequences_config['enabled'] or special_sequences_config['probability'] == 0:
        return text

    words = text.split(' ')
    if not words:
        return text

    # Récupérer les codes disponibles
    q_codes = list(morse_dicts.get("Q_CODES", {}).keys()) if special_sequences_config.get('qCodes', False) else []
    abbreviations = list(morse_dicts.get("ABBREVIATIONS", {}).keys()) if special_sequences_config.get('abbreviations', False) else []

    all_sequences = []
    if q_codes: 
        all_sequences.extend(q_codes)
    if abbreviations: 
        all_sequences.extend(abbreviations)

    if not all_sequences:
        return text

    # Insertion aléatoire des séquences
    new_words = []
    for word in words:
        new_words.append(word)
        if random.random() < special_sequences_config['probability']:
            insert_pos = random.choice([0, 1])  # Avant ou après le mot
            sequence = random.choice(all_sequences)
            if insert_pos == 0:
                new_words.insert(len(new_words) - 1, sequence)
            else:
                new_words.append(sequence)
    
    return " ".join(new_words)

def text_to_morse_audio(text_content, wpm, config):
    """
    CONVERSION TEXTE VERS AUDIO MORSE
    =================================
    Cœur du générateur : convertit une chaîne de texte en signal audio
    Morse avec tous les effets et variations réalistes.
    
    Arguments:
    - text_content: Texte à convertir
    - wpm: Vitesse en mots par minute
    - config: Configuration complète du générateur
    
    Retourne:
    - (audio_data, clean_text): Signal audio et texte réellement converti
    """
    # RÉCUPÉRATION DE LA CONFIGURATION
    audio_opts = config['audioOptions']
    sample_rate = audio_opts['sampleRate']
    morse_tone_freq = audio_opts['morseToneFreq'] 
    max_amplitude = audio_opts['maxAmplitude']
    human_var_min = audio_opts['humanVarMin']
    human_var_max = audio_opts['humanVarMax']

    # FUSION DES DICTIONNAIRES MORSE
    morse_code_dict = config['morseDictionaries']['MORSE_CODE_DICT']
    morse_accented_chars_dict = config['morseDictionaries']['MORSE_ACCENTED_CHARS_DICT']
    prosign_dict = config['morseDictionaries']['PROSIGN_DICT']
    
    # Les prosigns ont la priorité en cas de conflit
    full_morse_dict = {**morse_code_dict, **morse_accented_chars_dict, **prosign_dict}

    # CALCUL DES DURÉES MORSE SELON LA VITESSE
    dit_duration = get_morse_duration(wpm, "dot")
    dah_duration = get_morse_duration(wpm, "dash")  
    intra_char_space_duration = get_morse_duration(wpm, "intra_char_space")
    char_space_duration = get_morse_duration(wpm, "char_space")
    word_space_duration = get_morse_duration(wpm, "word_space")

    audio_segments = []  # Stockage de tous les segments audio
    clean_text_chars = []  # Caractères réellement convertis

    # TRAITEMENT MOT PAR MOT
    words = text_content.strip().split(' ')
    for i, word in enumerate(words):
        # TRAITEMENT CARACTÈRE PAR CARACTÈRE
        for j, char in enumerate(word):
            morse_char = None
            
            # RECHERCHE DU CODE MORSE (ORDRE DE PRIORITÉ)
            if char in prosign_dict:
                morse_char = prosign_dict[char]
            elif char in morse_accented_chars_dict:
                morse_char = morse_accented_chars_dict[char]
            elif char.upper() in morse_code_dict:
                morse_char = morse_code_dict[char.upper()]
            
            if morse_char:
                clean_text_chars.append(char)
                
                # CONVERSION DE CHAQUE POINT/TRAIT EN AUDIO
                for k, symbol in enumerate(morse_char):
                    # Configuration de la dérive de fréquence
                    freq_drift_enabled = audio_opts['advanced']['frequencyDrift']['enabled']
                    max_variation_hz = audio_opts['advanced']['frequencyDrift']['maxVariationHz']

                    if symbol == '.':  # POINT
                        duration = apply_human_variability(dit_duration, human_var_min, human_var_max)
                        tone = generate_tone(morse_tone_freq, duration, sample_rate, 
                                           max_amplitude, freq_drift_enabled, max_variation_hz)
                    elif symbol == '-':  # TRAIT  
                        duration = apply_human_variability(dah_duration, human_var_min, human_var_max)
                        tone = generate_tone(morse_tone_freq, duration, sample_rate,
                                           max_amplitude, freq_drift_enabled, max_variation_hz)
                    else:
                        tone = np.array([])  # Ne devrait pas arriver

                    audio_segments.append(tone)

                    # Espace entre points/traits du même caractère
                    if k < len(morse_char) - 1:
                        space_duration = apply_human_variability(intra_char_space_duration, 
                                                               human_var_min, human_var_max)
                        audio_segments.append(np.zeros(int(sample_rate * space_duration)))

            # Espace entre caractères d'un même mot
            if j < len(word) - 1:
                space_duration = apply_human_variability(char_space_duration, 
                                                       human_var_min, human_var_max)
                audio_segments.append(np.zeros(int(sample_rate * space_duration)))
        
        # Espace entre mots
        if i < len(words) - 1:
            space_duration = apply_human_variability(word_space_duration, 
                                                   human_var_min, human_var_max)
            audio_segments.append(np.zeros(int(sample_rate * space_duration)))

    # ASSEMBLAGE DU SIGNAL AUDIO COMPLET
    audio_data = np.concatenate(audio_segments) if audio_segments else np.array([])

    if len(audio_data) == 0:
        return np.zeros(int(sample_rate * 0.1)), ""  # Signal vide si aucun caractère valide

    # NORMALISATION INITIALE
    audio_data = audio_data / np.max(np.abs(audio_data)) * max_amplitude

    # APPLICATION DES EFFETS AVANCÉS
    # ==============================
    advanced = audio_opts['advanced']

    # EFFET DE FADING
    if advanced['fading']['enabled']:
        audio_data = apply_fading(audio_data, 
                                advanced['fading']['maxDepthDb'], 
                                advanced['fading']['speedVariation'], 
                                sample_rate)

    # DISTORSION
    distortion_types = advanced['distortion']['type']
    distortion_probability = advanced['distortion']['probability']

    if distortion_probability > 0 and distortion_types:
        # Gérer le cas où "none" est mélangé avec d'autres types
        if "none" in distortion_types and len(distortion_types) > 1:
            actual_dist_types = [dt for dt in distortion_types if dt != "none"]
            if actual_dist_types and random.random() < distortion_probability:
                dist_type = random.choice(actual_dist_types)
                audio_data = apply_distortion(audio_data, dist_type)
        elif distortion_types != ["none"]:  # Pas seulement "none"
            if random.random() < distortion_probability:
                dist_type = random.choice(distortion_types)
                audio_data = apply_distortion(audio_data, dist_type)

    # BRUIT IMPULSIONNEL
    if advanced['impulseNoise']['enabled']:
        audio_data = add_impulse_noise(audio_data, 
                                     advanced['impulseNoise']['probability'], 
                                     advanced['impulseNoise']['magnitude'])

    # AJOUT DU BRUIT DE FOND (TOUJOURS EN DERNIER)
    selected_noise_types = audio_opts['selectedNoiseTypes']
    selected_snr_levels = audio_opts['snrLevels']
    
    if selected_noise_types and selected_snr_levels:
        noise_type = random.choice(selected_noise_types)
        snr_db = random.choice(selected_snr_levels)
        audio_data = add_noise(audio_data, noise_type, snr_db, sample_rate)

    # NORMALISATION FINALE
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * max_amplitude
    
    clean_text = "".join(clean_text_chars)
    return audio_data, clean_text

def generate_dataset(config_path):
    """
    GÉNÉRATION COMPLÈTE DU DATASET
    ==============================
    Fonction principale qui orchestre la génération de milliers de fichiers
    audio et texte selon la configuration fournie.
    
    Arguments:
    - config_path: Chemin vers le fichier de configuration JSON
    """
    # CHARGEMENT DE LA CONFIGURATION
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    gen_opts = config['generationOptions']
    audio_opts = config['audioOptions'] 
    output_opts = config['outputOptions']
    morse_dicts = config['morseDictionaries']

    output_dir = output_opts['outputDir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialiser les dictionnaires additionnels si nécessaires
    morse_dicts["Q_CODES"] = morse_dicts.get("Q_CODES", {})
    morse_dicts["ABBREVIATIONS"] = morse_dicts.get("ABBREVIATIONS", {})

    # GESTION DES MÉTADONNÉES ET STATISTIQUES
    metadata_path = os.path.join(output_dir, 'metadata.json')
    stats_path = os.path.join(output_dir, 'statistics.json')

    # Charger les données existantes ou initialiser
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {"files": []}
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    else:
        stats = {
            "total_files": 0,
            "total_audio_files": 0, 
            "total_characters": 0,
            "total_words": 0,
            "total_duration_seconds": 0.0
        }

    # GÉNÉRATION DES FICHIERS
    start_number = find_next_file_number(output_dir)
    files_generated = 0

    print(f"Démarrage de la génération à partir du numéro {start_number:04d}...")
    
    # Barre de progression pour l'expérience utilisateur
    for i in tqdm(range(gen_opts['numTextFiles']), desc="Génération des fichiers"):
        # GÉNÉRATION DU CONTENU TEXTUEL
        chars_per_group = random.randint(5, 15) * gen_opts['groupSize']
        
        random_text = generate_random_text(chars_per_group, 
                                         gen_opts['selectedCharTypes'], 
                                         gen_opts['characterFrequency'],
                                         morse_dicts)
        
        # Ajouter des séquences spéciales
        random_text = add_special_sequences(random_text, gen_opts['specialSequences'], morse_dicts)

        if not random_text.strip():
            continue  # Ignorer les textes vides

        # GÉNÉRATION DES VARIANTES AUDIO
        for _ in range(gen_opts['numGroups']):
            current_file_number = start_number + files_generated
            
            # Sélection aléatoire des paramètres
            wpm = random.choice(audio_opts['selectedWPMs'])
            
            # GÉNÉRATION DE L'AUDIO
            audio_data, clean_text = text_to_morse_audio(random_text, wpm, config)

            if len(audio_data) > 0:
                # SAUVEGARDE DES FICHIERS
                base_filename = f"morse_{current_file_number:04d}"
                text_filepath = os.path.join(output_dir, f"{base_filename}.txt")
                audio_filepath = os.path.join(output_dir, f"{base_filename}.{output_opts['audioFormat']}")

                # Fichier texte
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(clean_text)

                # Fichier audio
                wavfile.write(audio_filepath, audio_opts['sampleRate'], 
                            audio_data.astype(np.float32))

                # MISE À JOUR DES STATISTIQUES
                files_generated += 1
                stats['total_files'] += 1
                stats['total_characters'] += len(clean_text)
                stats['total_words'] += len(clean_text.split(' '))
                stats['total_duration_seconds'] += len(audio_data) / audio_opts['sampleRate']

                # AJOUT AUX MÉTADONNÉES
                metadata['files'].append({
                    'id': current_file_number,
                    'text_file': f"{base_filename}.txt",
                    'audio_file': f"{base_filename}.{output_opts['audioFormat']}",
                    'original_text': random_text,
                    'clean_text_audio': clean_text,
                    'wpm': wpm,
                    'noise_type': random.choice(audio_opts['selectedNoiseTypes']) if audio_opts['selectedNoiseTypes'] else "none",
                    'snr_db': random.choice(audio_opts['snrLevels']) if audio_opts['snrLevels'] else "N/A",
                    'audio_duration': len(audio_data) / config['audioOptions']['sampleRate'],
                    'character_count': len(clean_text)
                })
                
                stats['total_audio_files'] += 1
    
    # SAUVEGARDE DES MÉTADONNÉES FINALES
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Sauvegarder aussi la configuration utilisée
    with open(os.path.join(output_dir, 'config_used.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # RAPPORT FINAL
    print(f"\n✅ Dataset généré avec succès dans '{output_dir}'")
    print(f"📊 Statistiques finales:")
    print(f"   - Nouveaux fichiers générés: {files_generated}")
    print(f"   - Fichiers texte total: {stats['total_files']}")
    print(f"   - Fichiers audio total: {stats['total_audio_files']}")
    print(f"   - Dernier fichier créé: morse_{start_number + files_generated - 1:04d}")
    print(f"   - Durée totale audio: {stats['total_duration_seconds']:.2f} secondes")

# =============================================================================
# EXÉCUTION DU PROGRAMME PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Configuration par défaut
        config_file = 'Morse_gen_config.json'
        print(f"Aucun fichier de configuration spécifié. Utilisation de '{config_file}' par défaut.")
        print("Pour spécifier un fichier: python morse_generator_incremental.py votre_config.json")

    if not os.path.exists(config_file):
        print(f"Erreur: Le fichier de configuration '{config_file}' n'existe pas.")
        sys.exit(1)
    
    generate_dataset(config_file)

# =============================================================================
# GUIDE D'UTILISATION POUR LES UTILISATEURS
# =============================================================================
"""
COMMENT UTILISER CE GÉNÉRATEUR:
===============================

1. PRÉREQUIS:
   - Python 3.7+ avec les bibliothèques:
     pip install numpy scipy tqdm matplotlib

2. CONFIGURATION:
   - Utiliser le générateur web (generateur.html) OU
   - Modifier manuellement le fichier Morse_gen_config.json OU
   - Créer votre propre fichier de configuration JSON

3. EXÉCUTION:
   python morse_generator_incremental.py [fichier_config.json]

4. RÉSULTATS:
   - Dossier "mon_dataset/" (ou selon votre config) créé
   - Paires de fichiers: morse_XXXX.wav + morse_XXXX.txt
   - Fichiers metadata.json et statistics.json
   - Fichier config_used.json (sauvegarde des paramètres)

STRUCTURE DE LA CONFIGURATION JSON:
==================================
{
  "generationOptions": {
    "numTextFiles": 100,           // Nombre de textes différents
    "groupSize": 1,                // Multiplicateur de longueur  
    "numGroups": 1,                // Variantes par texte
    "selectedCharTypes": [...],    // Types de caractères
    "characterFrequency": {...},   // Distribution des caractères
    "specialSequences": {...}      // Codes Q, abréviations
  },
  "audioOptions": {
    "selectedWPMs": [5, 10, 15],   // Vitesses en mots/minute
    "selectedNoiseTypes": [...],   // Types de bruit
    "snrLevels": [...],           // Niveaux signal/bruit
    "humanVarMin/Max": ...,       // Variabilité humaine
    "advanced": {                 // Effets avancés
      "frequencyDrift": {...},
      "fading": {...},
      "distortion": {...},
      "impulseNoise": {...}
    }
  },
  "morseDictionaries": {          // Dictionnaires de conversion
    "MORSE_CODE_DICT": {...},     // Alphabet + chiffres + ponctuation
    "MORSE_ACCENTED_CHARS_DICT": {...}, // Caractères accentués
    "PROSIGN_DICT": {...}         // Signaux de procédure
  }
}

CONSEILS D'UTILISATION:
=======================
- Commencez petit (100 fichiers) pour tester
- Utilisez plusieurs vitesses WPM pour la robustesse
- Le bruit "none" + SNR élevé = conditions idéales
- Le bruit "white" + SNR bas = conditions difficiles  
- Les effets avancés simulent des conditions réalistes
- Vérifiez le dossier de sortie avant de lancer une grosse génération
"""