"""
Gestionnaire de télémétrie de l'entraînement.
Agrège les métriques du système (fonctions de perte du VAE/RNN, score du Controller, latence d'exécution) et les sérialise vers Weights & Biases ou des fichiers locaux pour monitorer l'adaptation de l'IA sur la nouvelle map.
"""