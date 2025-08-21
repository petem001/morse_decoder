# =============================================================================
# PROGRAMME D'ENTRAÎNEMENT POUR RECONNAISSANCE VOCALE DU CODE MORSE
# =============================================================================
# Ce programme entraîne un modèle d'intelligence artificielle à reconnaître 
# le code Morse parlé et le convertir en texte
# =============================================================================

# IMPORTATION DES BIBLIOTHÈQUES NÉCESSAIRES
# ------------------------------------------
import os           # Pour naviguer dans les dossiers
import json         # Pour lire les fichiers de configuration
import numpy as np  # Pour les calculs mathématiques sur les tableaux
import tensorflow as tf  # Bibliothèque d'IA de Google
from tensorflow import keras
from sklearn.model_selection import train_test_split  # Pour diviser les données
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CONFIGURATION DU PROGRAMME
# ---------------------------
# Ces valeurs contrôlent comment le modèle sera entraîné
PREPROCESSED_DIR = "preprocessed_data"  # Dossier contenant les données audio préparées
MAX_SEQ_LENGTH = 2000   # Longueur maximale d'un échantillon audio (en frames)
BATCH_SIZE = 32         # Nombre d'échantillons traités en même temps
EPOCHS = 100            # Nombre de fois où on passe sur toutes les données
MODEL_SAVE_PATH = "morse_model.keras"  # Nom du fichier où sauvegarder le modèle

def load_data():
    """
    FONCTION DE CHARGEMENT DES DONNÉES
    ==================================
    Cette fonction lit tous les fichiers audio et leurs transcriptions,
    puis les prépare pour l'entraînement du modèle.
    
    Retourne:
    - X: Les données audio (spectrogrammes MFCC)
    - y: Les textes correspondants (labels)
    - input_lengths: Longueur de chaque audio
    - label_lengths: Longueur de chaque texte
    - char_to_int: Dictionnaire pour convertir lettres en chiffres
    """
    
    # CHARGEMENT DU FICHIER DE MÉTADONNÉES
    # -------------------------------------
    # Le fichier metadata.json contient la liste de tous les échantillons
    # et le dictionnaire de conversion caractère -> chiffre
    with open(os.path.join(PREPROCESSED_DIR, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # PRÉPARATION DU DICTIONNAIRE DE CARACTÈRES
    # ------------------------------------------
    # Le modèle ne comprend que les chiffres, on doit convertir les lettres
    char_to_int = metadata["char_map"]
    char_to_int['<PAD>'] = 0  # Token de remplissage pour égaliser les longueurs
    char_to_int['<BLANK>'] = len(char_to_int)  # Token vide pour l'algorithme CTC

    # INITIALISATION DES LISTES DE DONNÉES
    # -------------------------------------
    X = []              # Contiendra les données audio
    y = []              # Contiendra les textes
    input_lengths = []  # Longueur de chaque audio
    label_lengths = []  # Longueur de chaque texte

    # CHARGEMENT DE CHAQUE ÉCHANTILLON
    # ---------------------------------
    for item in metadata["files"]:
        try:
            # Charger le fichier audio préprocessé (MFCC)
            mfcc = np.load(os.path.join(PREPROCESSED_DIR, "mfccs", item["mfcc_file"]))
            text_label = item["text_label"]
            
            # VALIDATION DE LA QUALITÉ DES DONNÉES
            # ------------------------------------
            # On s'assure que l'échantillon est utilisable
            seq_length = min(mfcc.shape[0], MAX_SEQ_LENGTH)
            if seq_length == 0 or len(text_label) == 0:
                continue  # Ignorer les échantillons vides
                
            # VÉRIFICATION POUR L'ALGORITHME CTC
            # -----------------------------------
            # L'audio doit être plus long que le texte pour que CTC fonctionne
            if seq_length <= len(text_label):
                continue
                
            # AJOUT DES DONNÉES VALIDES
            # -------------------------
            X.append(mfcc[:seq_length])  # Tronquer si trop long
            y.append([char_to_int[c] for c in text_label])  # Convertir lettres en chiffres
            input_lengths.append(seq_length)
            label_lengths.append(len(text_label))
            
        except Exception as e:
            print(f"Erreur lors du chargement de {item['mfcc_file']}: {str(e)}")
            continue

    # VÉRIFICATION QU'ON A DES DONNÉES
    # --------------------------------
    if len(X) == 0:
        raise ValueError("Aucun échantillon valide trouvé!")

    # ÉGALISATION DE LA TAILLE DES DONNÉES (PADDING)
    # -----------------------------------------------
    # Tous les échantillons doivent avoir la même taille pour l'entraînement
    max_label_length = max(label_lengths)
    X_padded = np.zeros((len(X), MAX_SEQ_LENGTH, 39), dtype=np.float32)  # 39 = nb de features MFCC
    y_padded = np.zeros((len(y), max_label_length), dtype=np.int32)
    
    # Remplir les tableaux avec nos données
    for i, (mfcc, label) in enumerate(zip(X, y)):
        X_padded[i, :len(mfcc)] = mfcc      # Copier l'audio
        y_padded[i, :len(label)] = label    # Copier le texte

    return X_padded, y_padded, np.array(input_lengths), np.array(label_lengths), char_to_int

def build_model(input_shape, num_classes):
    """
    CONSTRUCTION DE L'ARCHITECTURE DU MODÈLE
    ========================================
    Cette fonction crée le "cerveau" de notre IA qui apprendra à reconnaître
    le code Morse. Le modèle utilise l'algorithme CTC (Connectionist Temporal
    Classification) qui est spécialement conçu pour aligner audio et texte.
    
    Le modèle a plusieurs couches:
    1. Couches convolutionnelles : extraient les caractéristiques de l'audio
    2. Couches LSTM bidirectionnelles : analysent les séquences dans les deux sens
    3. Couche de sortie : produit les probabilités pour chaque caractère
    """
    
    # DÉFINITION DES ENTRÉES DU MODÈLE
    # ---------------------------------
    input_mfcc = Input(shape=input_shape, name='input_mfcc')        # Audio d'entrée
    input_length = Input(shape=(1,), name='input_length', dtype='int32')  # Longueur audio
    label_length = Input(shape=(1,), name='label_length', dtype='int32')  # Longueur texte
    labels = Input(shape=(None,), name='labels', dtype='int32')     # Texte cible
    
    # COUCHES D'EXTRACTION DE CARACTÉRISTIQUES
    # -----------------------------------------
    # Ces couches analysent l'audio pour détecter les motifs du Morse
    
    # Première couche convolutionnelle
    x = Conv1D(128, 3, padding='same', activation='relu')(input_mfcc)
    x = BatchNormalization()(x)  # Normalise les données pour stabiliser l'entraînement
    x = Dropout(0.3)(x)         # Évite le sur-apprentissage en "oubliant" 30% des connexions
    
    # Deuxième couche convolutionnelle
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # COUCHES DE TRAITEMENT SÉQUENTIEL
    # ---------------------------------
    # Ces couches analysent l'audio dans le temps (passé et futur)
    
    # Première couche LSTM bidirectionnelle (128 neurones dans chaque direction)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Deuxième couche LSTM bidirectionnelle (64 neurones dans chaque direction)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # COUCHE DE SORTIE
    # ----------------
    # Produit les probabilités pour chaque caractère (+1 pour le token "blank")
    predictions = Dense(num_classes + 1, activation='softmax', name='predictions')(x)
    
    # COUCHE DE PERTE CTC
    # -------------------
    # Calcule l'erreur entre les prédictions et la vraie transcription
    ctc_loss = keras.layers.Lambda(
        lambda args: keras.backend.ctc_batch_cost(args[0], args[1], args[2], args[3]),
        output_shape=(1,),
        name='ctc_loss'
    )([labels, predictions, input_length, label_length])
    
    # CRÉATION DES MODÈLES
    # --------------------
    # Modèle d'entraînement (avec calcul de perte)
    training_model = Model(
        inputs=[input_mfcc, labels, input_length, label_length],
        outputs=ctc_loss
    )
    
    # Modèle de prédiction (pour utilisation après entraînement)
    prediction_model = Model(inputs=input_mfcc, outputs=predictions)
    
    return training_model, prediction_model

class DataGenerator(keras.utils.Sequence):
    """
    GÉNÉRATEUR DE DONNÉES PERSONNALISÉ
    ==================================
    Cette classe organise les données en "lots" (batches) pour l'entraînement.
    Elle permet de traiter de gros volumes de données sans saturer la mémoire.
    
    Avantages:
    - Mélange les données à chaque époque
    - Traite les données par petits groupes
    - Prépare automatiquement le format requis par CTC
    """
    
    def __init__(self, X, y, input_lengths, label_lengths, batch_size=32, shuffle=True):
        """
        INITIALISATION DU GÉNÉRATEUR
        ----------------------------
        """
        self.X = X                          # Données audio
        self.y = y                          # Textes correspondants
        self.input_lengths = input_lengths   # Longueurs des audios
        self.label_lengths = label_lengths   # Longueurs des textes
        self.batch_size = batch_size        # Taille des lots
        self.shuffle = shuffle              # Mélanger ou pas
        self.indices = np.arange(len(self.X))  # Index des échantillons
        self.on_epoch_end()                 # Mélange initial
    
    def __len__(self):
        """Retourne le nombre de lots par époque"""
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        """
        GÉNÉRATION D'UN LOT DE DONNÉES
        ------------------------------
        Cette méthode est appelée pour chaque lot pendant l'entraînement
        """
        # Sélectionner les indices pour ce lot
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Récupérer les données correspondantes
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        batch_input_lengths = self.input_lengths[batch_indices]
        batch_label_lengths = self.label_lengths[batch_indices]
        
        # PRÉPARATION POUR LE MODÈLE CTC
        # -------------------------------
        # Le modèle CTC a besoin d'un format spécial
        inputs = {
            'input_mfcc': batch_X,
            'labels': batch_y,
            'input_length': batch_input_lengths.reshape(-1, 1),
            'label_length': batch_label_lengths.reshape(-1, 1)
        }
        
        # CTC calcule sa propre perte, on donne des cibles factices
        targets = np.zeros((len(batch_indices), 1))
        
        return inputs, targets
    
    def on_epoch_end(self):
        """Mélange les données à la fin de chaque époque"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# =============================================================================
# PROGRAMME PRINCIPAL D'ENTRAÎNEMENT
# =============================================================================

if __name__ == "__main__":
    
    # ÉTAPE 1 : CHARGEMENT ET VALIDATION DES DONNÉES
    # ===============================================
    print("📥 Chargement des données...")
    X, y, input_lengths, label_lengths, char_to_int = load_data()
    
    # Affichage des statistiques
    print(f"✅ {len(X)} échantillons valides chargés")
    print(f"📚 Vocabulaire: {len(char_to_int)} caractères")
    print(f"🎵 Longueur audio: {input_lengths.min()} - {input_lengths.max()} frames")
    print(f"📝 Longueur texte: {label_lengths.min()} - {label_lengths.max()} caractères")
    
    # ÉTAPE 2 : DIVISION ENTRAÎNEMENT/VALIDATION
    # ===========================================
    # On garde 80% des données pour l'entraînement, 20% pour tester
    print("\n🔀 Division des données...")
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Séparation des données
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    input_lengths_train, input_lengths_val = input_lengths[train_idx], input_lengths[val_idx]
    label_lengths_train, label_lengths_val = label_lengths[train_idx], label_lengths[val_idx]
    
    # ÉTAPE 3 : CRÉATION DES GÉNÉRATEURS DE DONNÉES
    # ==============================================
    print("🏭 Création des générateurs de données...")
    
    # Générateur pour l'entraînement (mélange activé)
    train_generator = DataGenerator(
        X_train, y_train, input_lengths_train, label_lengths_train,
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    # Générateur pour la validation (pas de mélange)
    val_generator = DataGenerator(
        X_val, y_val, input_lengths_val, label_lengths_val,
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    # ÉTAPE 4 : CONSTRUCTION DU MODÈLE
    # =================================
    print("🏗️ Construction du modèle d'IA...")
    training_model, prediction_model = build_model((MAX_SEQ_LENGTH, 39), len(char_to_int))
    
    # ÉTAPE 5 : COMPILATION DU MODÈLE D'ENTRAÎNEMENT
    # ===============================================
    # Configuration de l'optimiseur et de la fonction de perte
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Algorithme d'optimisation
        loss={'ctc_loss': lambda y_true, y_pred: y_pred}       # Perte CTC déjà calculée
    )
    
    # Affichage des informations du modèle
    print(f"\n🗂️ Architecture du modèle:")
    print(f"📊 Échantillons d'entraînement: {len(X_train)}")
    print(f"🧪 Échantillons de validation: {len(X_val)}")
    print(f"🔤 Taille du vocabulaire: {len(char_to_int)} caractères")
    training_model.summary()  # Affiche la structure détaillée
    
    # ÉTAPE 6 : CONFIGURATION DES CALLBACKS
    # ======================================
    # Les callbacks sont des fonctions appelées pendant l'entraînement
    callbacks = [
        # Arrêt précoce si le modèle ne s'améliore plus
        EarlyStopping(
            monitor='val_loss',           # Surveiller la perte de validation
            patience=10,                  # Attendre 10 époques sans amélioration
            restore_best_weights=True     # Restaurer les meilleurs poids
        ),
        
        # Sauvegarde automatique du meilleur modèle
        ModelCheckpoint(
            MODEL_SAVE_PATH.replace('.keras', '_best.keras'),  # Nom du fichier
            monitor='val_loss',           # Critère de sélection
            save_best_only=True,         # Sauvegarder seulement le meilleur
            mode='min',                  # Minimiser la perte
            save_weights_only=False      # Sauvegarder le modèle complet
        )
    ]
    
    # ÉTAPE 7 : LANCEMENT DE L'ENTRAÎNEMENT
    # ======================================
    print("🚀 Début de l'entraînement...")
    print("   (Cela peut prendre plusieurs heures selon votre matériel)")
    
    # Entraînement proprement dit
    history = training_model.fit(
        train_generator,              # Données d'entraînement
        validation_data=val_generator, # Données de validation
        epochs=EPOCHS,               # Nombre d'époques maximum
        callbacks=callbacks,         # Fonctions de callback
        verbose=1                    # Affichage détaillé
    )
    
    # ÉTAPE 8 : SAUVEGARDE FINALE
    # ============================
    print("\n💾 Sauvegarde du modèle...")
    
    # Sauvegarder le modèle de prédiction
    prediction_model.save(MODEL_SAVE_PATH)
    print(f"✅ Modèle de prédiction sauvegardé: {MODEL_SAVE_PATH}")
    
    # Sauvegarder l'historique d'entraînement pour analyse
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f, indent=2)
    print("📊 Historique d'entraînement sauvegardé: training_history.json")
    
    print("\n🎉 Entraînement terminé avec succès!")
    print("   Votre modèle est prêt à reconnaître le code Morse!")

# =============================================================================
# RÉSUMÉ DU FONCTIONNEMENT
# =============================================================================
"""
Ce programme fonctionne en plusieurs étapes:

1. PRÉPARATION DES DONNÉES
   - Charge les fichiers audio préprocessés (MFCC)
   - Charge les transcriptions textuelles correspondantes
   - Convertit les lettres en nombres pour le modèle
   - Égalise la taille de tous les échantillons

2. ARCHITECTURE DU MODÈLE
   - Couches convolutionnelles: détectent les motifs dans l'audio
   - Couches LSTM: analysent les séquences temporelles
   - Algorithme CTC: aligne automatiquement audio et texte

3. ENTRAÎNEMENT
   - Le modèle apprend à associer les sons Morse aux lettres
   - Utilise les données de validation pour éviter le sur-apprentissage
   - Sauvegarde automatiquement le meilleur modèle

4. RÉSULTAT
   - Un modèle capable de transcrire du code Morse parlé en texte
   - Fichiers de sauvegarde pour utilisation ultérieure
"""