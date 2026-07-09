"""
Point d'entrée de LTM-AI.
Instancie la configuration globale, lance les processus asynchrones (interface utilisateur, logger) et
démarre l'orchestrateur.
"""




"""
import torch.multiprocessing as mp
from src.utils.config import load_config
from src.orchestrator import Orchestrator
from src.ui.dashboard import LTM_Dashboard

def main():
    # 1. Éviter les problèmes de démarrage multi-processus sous Windows/Linux
    mp.set_start_method('spawn', force=True)

    # 2. Chargement de la configuration
    config = load_config("config/default.yaml")

    # 3. Création des signaux et canaux de communication partagés
    stop_event = mp.Event()
    mode_queue = mp.Queue()       # Pour changer les 3 modes d'entraînement
    experience_queue = mp.Queue() # L'Acteur y met les images, l'Apprenti les lit
    weights_queue = mp.Queue()    # L'Apprenti y met les nouveaux poids, l'Acteur les lit

    # 4. Instanciation de l'Orchestrateur
    orchestrator = Orchestrator(
        config=config,
        stop_event=stop_event,
        mode_queue=mode_queue,
        experience_queue=experience_queue,
        weights_queue=weights_queue
    )

    try:
        # 5. Démarrage des processus enfants (Trackmania + Entraînement)
        orchestrator.start_workers()

        # 6. Lancement de l'interface graphique dans le thread principal
        # On passe l'orchestrateur à l'UI pour qu'elle puisse lui envoyer des ordres
        app = LTM_Dashboard(orchestrator)
        app.run() # <--- Le programme "bloque" ici tant que l'UI est ouverte

    except KeyboardInterrupt:
        # Gestion si on coupe via le terminal (Ctrl+C)
        print("\nInterruption détectée. Fermeture en cours...")

    finally:
        # 7. Séquence d'arrêt sécurisée (déclenchée à la fermeture de l'UI ou Ctrl+C)
        print("Envoi du signal d'arrêt aux processus...")
        stop_event.set() # Prévient l'Acteur et l'Apprenti de s'arrêter
        
        # L'orchestrateur s'assure que les processus sauvegardent et se ferment
        orchestrator.join_workers() 
        print("Fermeture complète du projet LTM-AI. Modèles sauvegardés.")

if __name__ == "__main__":
    main()
"""