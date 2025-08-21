# Guide d'Utilisation Complet - Système de Reconnaissance CODE Morse

## Vue d'ensemble du projet

Ce projet vous permet de créer un système d'intelligence artificielle capable de reconnaître et transcrire automatiquement le code Morse. Le processus complet se déroule en 4 étapes principales :

1. **Configuration** : Définir les paramètres de génération
2. **Génération** : Créer un dataset d'audio Morse synthétique  
3. **Prétraitement** : Transformer l'audio en données exploitables
4. **Entraînement** : Apprendre à l'IA à reconnaître le Morse

---

## Étape 1 : Configuration du Dataset

### Option A : Interface Web (Recommandée pour débutants)

**Fichier utilisé** : `generateur.html`

1. Ouvrez le fichier `generateur.html` dans votre navigateur web
2. Configurez vos paramètres via l'interface graphique :
   - **Nombre de fichiers** : Commencez par 50-100 pour tester
   - **Types de caractères** : Alphabet, chiffres, ponctuation selon vos besoins
   - **Vitesses WPM** : 5-15 WPM pour débuter (plus lent = plus facile)
   - **Conditions audio** : Commencez sans bruit, ajoutez progressivement
3. Téléchargez le fichier `Morse_gen_config.json` généré

### Option B : Modification manuelle

**Fichier utilisé** : `Morse_gen_config.json`

Modifiez directement le fichier JSON selon vos besoins. Voici les paramètres clés :

```json
{
  "generationOptions": {
    "numTextFiles": 100,        // Nombre de textes différents à générer
    "selectedCharTypes": ["alphabet", "numbers"],  // Types de caractères
    "selectedWPMs": [5, 10, 15] // Vitesses en mots par minute
  }
}
```

---

## Étape 2 : Génération du Dataset Audio

**Fichier utilisé** : `morse_generator_incremental.py`

### Installation des dépendances
```bash
pip install numpy scipy tqdm matplotlib
```

### Exécution
```bash
# Utilisation du fichier de config par défaut
python morse_generator_incremental.py

# Ou spécifier un fichier de configuration
python morse_generator_incremental.py ma_config.json
```

### Résultats attendus
- Dossier `mon_dataset/` créé contenant :
  - Fichiers audio : `morse_0000.wav`, `morse_0001.wav`, etc.
  - Transcriptions : `morse_0000.txt`, `morse_0001.txt`, etc.
  - `metadata.json` : Informations sur tous les fichiers
  - `statistics.json` : Statistiques du dataset

### Vérification de la qualité
- Écoutez quelques fichiers audio pour vérifier la qualité
- Vérifiez que les transcriptions correspondent à l'audio
- Consultez les statistiques pour confirmer la diversité

---

## Étape 3 : Prétraitement des Données

**Fichier utilisé** : `preprocess.py`

### Installation des dépendances supplémentaires
```bash
pip install librosa
```

### Exécution
```bash
python preprocess.py
```

Le script :
1. Lit tous les fichiers `.wav` et `.txt` du dossier `mon_dataset/`
2. Extrait les caractéristiques MFCC (représentation numérique de l'audio)
3. Crée le vocabulaire (association caractère ↔ numéro)
4. Sauvegarde tout dans `preprocessed_data/`

### Résultats attendus
- Dossier `preprocessed_data/` créé contenant :
  - `mfccs/` : Fichiers `.npy` avec les caractéristiques audio
  - `metadata.json` : Informations et vocabulaire

---

## Étape 4 : Entraînement du Modèle

**Fichier utilisé** : `trainv10.py`

### Installation des dépendances d'IA
```bash
pip install tensorflow scikit-learn
```

### Exécution
```bash
python trainv10.py
```

Le processus d'entraînement :
1. Charge les données prétraitées
2. Divise en ensemble d'entraînement (80%) et validation (20%)
3. Entraîne le modèle d'IA sur plusieurs époques
4. Sauvegarde automatiquement le meilleur modèle

### Résultats attendus
- `morse_model.keras` : Modèle entraîné prêt à utiliser
- `training_history.json` : Historique de l'entraînement
- Affichage des métriques d'entraînement en temps réel

---

## Analyse des Résultats d'Entraînement

**Fichier d'exemple** : `training_history.json`

Votre historique d'entraînement montre une évolution positive :
- **Perte d'entraînement** : De 4455 → 39.3 (diminution continue)
- **Perte de validation** : De 4374 → 40.6 (convergence stable)

Cette courbe indique :
- ✅ Le modèle apprend correctement
- ✅ Pas de sur-apprentissage visible
- ✅ Convergence stable après ~50 époques

---

## Conseils Pratiques par Niveau

### Débutant
1. **Commencez simple** :
   - 50-100 fichiers
   - Seulement alphabet + chiffres
   - Une seule vitesse (5 WPM)
   - Pas de bruit

2. **Vérifiez chaque étape** :
   - Écoutez quelques fichiers générés
   - Vérifiez que le prétraitement fonctionne
   - Surveillez l'entraînement

### Intermédiaire
1. **Augmentez la complexité** :
   - 500-1000 fichiers
   - Ajoutez la ponctuation
   - Plusieurs vitesses (5, 10, 15 WPM)
   - Bruit léger

2. **Optimisez les paramètres** :
   - Ajustez la taille des lots (BATCH_SIZE)
   - Modifiez le taux d'apprentissage
   - Testez différentes architectures

### Avancé
1. **Dataset complexe** :
   - 1000+ fichiers
   - Tous types de caractères
   - Gamme complète de vitesses
   - Effets réalistes (bruit, fading, distorsion)

2. **Fine-tuning** :
   - Hyperparamètres optimisés
   - Augmentation de données
   - Validation croisée

---

## Dépannage Courant

### Problème : "Aucun échantillon valide trouvé"
**Solution** :
- Vérifiez que les fichiers .wav et .txt existent par paires
- Contrôlez que les transcriptions ne sont pas vides
- Ajustez MAX_DURATION dans preprocess.py si nécessaire

### Problème : Mémoire insuffisante pendant l'entraînement
**Solutions** :
- Réduire BATCH_SIZE dans trainv10.py
- Diminuer MAX_SEQ_LENGTH
- Utiliser moins de fichiers d'entraînement

### Problème : Le modèle ne converge pas
**Solutions** :
- Augmenter le nombre d'époques
- Réduire le taux d'apprentissage
- Simplifier le dataset (moins de bruit, vitesse constante)

### Problème : Perte qui stagne
**Solutions** :
- Augmenter la taille du dataset
- Vérifier la qualité des données
- Ajuster l'architecture du modèle

---

## Structure des Fichiers Recommandée

```
projet_morse/
├── generateur.html              # Interface de configuration
├── Morse_gen_config.json        # Configuration générée
├── morse_generator_incremental.py  # Générateur de dataset
├── preprocess.py                # Prétraitement
├── trainv10.py                  # Entraînement
├── mon_dataset/                 # Dataset brut généré
│   ├── morse_0000.wav           # Fichiers audio
│   ├── morse_0000.txt           # Transcriptions
│   ├── metadata.json            # Informations des fichiers
│   └── statistics.json          # Statistiques du dataset
├── preprocessed_data/           # Données préparées pour l'IA
│   ├── mfccs/                   # Caractéristiques audio extraites
│   └── metadata.json            # Vocabulaire et config
├── morse_model.keras            # Modèle IA entraîné
├── training_history.json        # Historique d'entraînement
└── README.md                    # Ce guide
```

---

## Workflow Linéaire Complet

### Phase 1 : Préparation (5-10 minutes)
1. **Téléchargez tous les fichiers** du projet
2. **Installez Python 3.7+** si pas déjà fait
3. **Installez les dépendances** :
   ```bash
   pip install numpy scipy tqdm matplotlib librosa tensorflow scikit-learn
   ```

### Phase 2 : Configuration (2-5 minutes)
1. **Ouvrez `generateur.html`** dans votre navigateur
2. **Configurez vos paramètres** selon vos besoins
3. **Téléchargez le fichier JSON** généré
4. **Renommez-le** en `Morse_gen_config.json` dans le dossier du projet

### Phase 3 : Génération du Dataset (10-60 minutes selon la taille)
1. **Exécutez** : `python morse_generator_incremental.py`
2. **Attendez la fin** de la génération (barre de progression)
3. **Vérifiez** que le dossier `mon_dataset/` est créé avec des fichiers
4. **Testez** en écoutant quelques fichiers .wav

### Phase 4 : Prétraitement (5-20 minutes selon la taille)
1. **Exécutez** : `python preprocess.py`
2. **Attendez l'extraction** des caractéristiques MFCC
3. **Vérifiez** que le dossier `preprocessed_data/` est créé
4. **Contrôlez** les statistiques affichées

### Phase 5 : Entraînement (30 minutes à plusieurs heures)
1. **Exécutez** : `python trainv10.py`
2. **Surveillez** l'évolution des métriques
3. **Attendez** la fin automatique (early stopping)
4. **Vérifiez** que `morse_model.keras` est créé

---

## Estimation des Temps de Traitement

| Taille Dataset | Génération | Prétraitement | Entraînement | Total |
|----------------|------------|---------------|--------------|-------|
| 50 fichiers    | 2 min      | 1 min         | 15 min       | 18 min |
| 100 fichiers   | 5 min      | 2 min         | 30 min       | 37 min |
| 500 fichiers   | 15 min     | 8 min         | 2h           | 2h23  |
| 1000 fichiers  | 30 min     | 15 min        | 4h           | 4h45  |

*Temps approximatifs sur un ordinateur standard*

---

## Validation de la Qualité

### Après Génération
- **Écoutez** plusieurs fichiers audio
- **Vérifiez** que le Morse correspond aux transcriptions
- **Contrôlez** la diversité des vitesses et conditions

### Après Prétraitement
- **Vérifiez** le nombre de caractères uniques dans le vocabulaire
- **Contrôlez** que tous les fichiers ont été traités
- **Validez** les dimensions des données MFCC

### Après Entraînement
- **Analysez** les courbes de perte (train/validation)
- **Vérifiez** la convergence du modèle
- **Testez** sur quelques échantillons manuellement

---

## Utilisation du Modèle Entraîné

Une fois l'entraînement terminé, vous pouvez utiliser votre modèle :

```python
import tensorflow as tf
import numpy as np

# Charger le modèle
model = tf.keras.models.load_model('morse_model.keras')

# Prédire sur un nouveau fichier audio (après prétraitement MFCC)
# prediction = model.predict(mfcc_features)
```

---

## Amélioration Continue

### Pour Améliorer la Précision
1. **Augmentez la taille** du dataset
2. **Diversifiez les conditions** (vitesses, bruits)
3. **Ajustez l'architecture** du réseau de neurones
4. **Optimisez les hyperparamètres**

### Pour Réduire les Temps
1. **Utilisez un GPU** pour l'entraînement
2. **Réduisez la résolution** audio (sample rate)
3. **Diminuez la complexité** du modèle
4. **Optimisez les paramètres** de preprocessing

---

## Ressources Additionnelles

### Documentation Technique
- **TensorFlow** : https://tensorflow.org/guide
- **Librosa** : https://librosa.org/doc/stable/
- **Code Morse** : Standards ITU-R M.1677

### Communauté
- Forums de radio amateur pour les standards Morse
- Documentation sur les algorithmes CTC
- Ressources sur le traitement du signal audio

---

## Licence et Distribution

Ce projet est conçu pour être partagé et modifié. Lorsque vous distribuez ces fichiers :

1. **Incluez ce guide** pour aider les nouveaux utilisateurs
2. **Documentez vos modifications** si vous changez le code
3. **Partagez vos résultats** avec la communauté
4. **Respectez les dépendances** des bibliothèques tierces

---

## Support et Contact

Si vous rencontrez des difficultés :

1. **Vérifiez ce guide** en détail
2. **Consultez les messages d'erreur** attentivement  
3. **Testez avec un dataset plus petit** d'abord
4. **Documentez votre configuration** système pour le dépannage

Le système a été testé avec succès sur Windows, Linux et macOS avec Python 3.7-3.11.
