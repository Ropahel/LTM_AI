"""
Pipeline de perception visuelle bas niveau.
Capture les frames de la fenêtre Trackmania avec une latence minimale (API graphique),
applique les transformations géométriques (recadrage de l'UI) et normalise les tenseurs d'images pour la suite des processus.
"""
import dxcam
import cv2
import numpy as np

class ScreenCapture:

    def __init__(self,
                 target_fps: int = 30,
                 resize_dim: tuple = (256, 256)) -> None:

        self.target_fps = target_fps
        self.resize_dim = resize_dim

        #Initialisation de dxcam en RGB
        self.camera = dxcam.create(output_idx=0, output_color="RGB")
        self.is_capturing = False


    def start(self):
        """Démarre la capture d'écran en arrière plan"""
        
        self.camera.start(target_fps=self.target_fps)
        self.is_capturing = True
        print(f"Capture démarrée à {self.target_fps} FPS max.")


    def stop(self):
        """Stop la capture d'écran en arrière plan"""
        
        if self.is_capturing:
            self.camera.stop()
            self.is_capturing = False
            print("Capture stoppée.")


    def get_last_frame(self):
        """Récupère la dernière frame prise en capture d'écran"""
        
        if not self.is_capturing:
            print("Pas possible de récupérer une image quand la caméra est éteinte")
            return None

        frame = self.camera.get_latest_frame()

        if frame is None:
            return None

        return frame


    def preprocess_car(self, frame, grayscale=False):
        """
        Pré-traitement de l'image pour ensuite être utilisable par les modèles pour l'état de la voiture.
        On effectue:
            - Redimensionnement : Selon self.resize_dim (ex: 256x256)
            - Normalisation (0.0 à 1.0)

        On garde ici toute l'image et on ne masque rien.
        """
        #Redimensionnement 
        resized_frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_AREA)
        
        if grayscale:
            #Convertit de RGB vers Gris
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
            
            #OpenCV supprime la dimension des canaux quand il passe en gris (H, W).
            #On la rajoute manuellement pour que le tenseur soit (H, W, 1)
            resized_frame = np.expand_dims(resized_frame, axis=-1)

        #Normalisation
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        return normalized_frame


    def preprocess_map(self, frame, grayscale=False):
        """
        Pré-traitement de l'image pour ensuite être utilisable par les modèles pour la map.
        On effectue:
            - Redimensionnement
            - On cache la voiture
            - Normalisation

        Pour ce preprocessement on va masquer la partie avec la voiture en y mettant un bloc noir
        """
        #Redimensionnement
        resized_frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_AREA)

        #Application du cache noir (Masquage)
        h, w = self.resize_dim
        
        #Définition de la zone de la voiture
        #Ici on suppose que la voiture est au centre-bas de l'écran.
        x_min = int(w * 0.35)  #35% de la largeur (à partir de la gauche)
        x_max = int(w * 0.65)  #65% de la largeur
        y_min = int(h * 0.55)  #60% de la hauteur (en partant du haut)
        y_max = int(h * 0.90)              #Jusqu'en bas (100%)

        #On met tous les pixels de cette zone à 0 (Noir). 
        resized_frame[y_min:y_max, x_min:x_max] = 0

        #Passage en niveaux de gris si demandé
        if grayscale:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2GRAY)
            resized_frame = np.expand_dims(resized_frame, axis=-1)

        #Normalisationp
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        return normalized_frame


#Test
if __name__ == "__main__":
    camera = ScreenCapture(target_fps=10, resize_dim=(256, 256))
    camera.start()

    print("Test de capture lancé. Clique sur la fenêtre vidéo et appuie sur la touche 'p' pour quitter.")

    # 1. Création d'une fenêtre redimensionnable AVANT la boucle
    nom_fenetre = "Vue de l'IA (Agrandie)"
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
    
    # 2. On force la taille d'affichage pour toi (par exemple 800x800 pixels)
    # L'image de base restera en 256x256 dans la mémoire de l'IA
    cv2.resizeWindow(nom_fenetre, 800, 800)

    try:
        while True:
            frame_brute = camera.get_last_frame()

            if frame_brute is not None:
                frame_pretraitee = camera.preprocess_map(frame_brute)
                frame_affichage = cv2.cvtColor(frame_pretraitee, cv2.COLOR_RGB2BGR)

                # 3. Affichage dans la fenêtre qu'on a configurée
                cv2.imshow(nom_fenetre, frame_affichage)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                print("\nTouche 'p' détectée. Arrêt normal.")
                break

    except KeyboardInterrupt:
        print("\nArrêt forcé via le terminal.")

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Test terminé et ressources libérées.")