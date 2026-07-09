"""
Interface d'actuation et de retour d'état pour Trackmania 2020.
Traduit les sorties du Controller en commandes d'entrée virtuelles (direction, accélération, frein)
au niveau de l'OS, et extrait la télémétrie du jeu (vitesse, position, checkpoints) pour formuler le signal de récompense continu.
"""