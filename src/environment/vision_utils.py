"""
Pipeline de perception visuelle bas niveau.
Capture les frames de la fenêtre Trackmania avec une latence minimale (API graphique),
applique les transformations géométriques (recadrage de l'UI) et normalise les tenseurs d'images pour la suite des processus.
"""
import dxcam
import cv2
import numpy as np



class ScreenCapture:
    pass