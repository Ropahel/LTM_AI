from vision_utils import ScreenCapture
import socket
import threading
import json
import time


class TMWrapper:
    def __init__(self, use_camera=True):
        # Camera optionnelle
        self.use_camera = use_camera
        self.camera = ScreenCapture(target_fps=10, resize_dim=(256, 256)) if use_camera else None

        # Etat interne
        self.current_telemetry = {
            "speed": 0.0,
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        self.connected = False
        self.has_telemetry = False

        self.telemetry_thread = threading.Thread(
            target=self.start_telemetry_listener,
            daemon=True,
        )
        self.telemetry_thread.start()


    def start_telemetry_listener(self):
        """Écoute OpenPlanet via TCP et décode les messages avec un header fixe de 4 octets."""
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp_socket.bind(("127.0.0.1", 9000))
        tcp_socket.listen(1)

        print("TMWrapper : En attente de la connexion d'OpenPlanet sur le port 9000...")

        while True:
            conn, addr = tcp_socket.accept()
            self.connected = True
            self.has_telemetry = False
            print(f"TMWrapper : OpenPlanet connecté depuis {addr} !")

            try:
                buffer = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk

                    # Découpage par '\n' = fin de message
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line:
                            continue

                        # DEBUG : voir la ligne brute une fois
                        # print("RAW LINE:", line)

                        # 1) Ignore les lignes trop courtes
                        if len(line) <= 4:
                            continue

                        # 2) Enlève systématiquement les 4 premiers octets (header)
                        json_bytes = line[4:]

                        try:
                            text = json_bytes.decode("utf-8")
                        except UnicodeDecodeError as e:
                            print(f"TMWrapper : ligne non décodable UTF-8, ignorée : {line!r} ({e})")
                            continue

                        text_stripped = text.strip()
                        if not text_stripped:
                            continue

                        # 3) Essaye de parser en JSON directement
                        try:
                            data = json.loads(text_stripped)
                        except json.JSONDecodeError as e:
                            print(f"TMWrapper : JSON invalide : {text_stripped!r} ({e})")
                            continue

                        # 4) Mise à jour de l'état interne
                        self.current_telemetry = data
                        self.has_telemetry = True

            except Exception as e:
                print(f"TMWrapper : Erreur de flux télémétrie - {e}")
            finally:
                conn.close()
                self.connected = False
                print("TMWrapper : Connexion perdue, en attente de reconnexion...")


    def step_loop_action(self, action):
        if self.use_camera and self.camera is not None:
            obs = self.camera.get_last_frame()
        else:
            obs = None

        state = self.current_telemetry
        return obs, state


if __name__ == "__main__":
    wrapper = TMWrapper(use_camera=False)

    print("Démarrage du test d'écoute. Lance Trackmania avec le plugin OpenPlanet actif.")

    try:
        while True:
            time.sleep(1)

            if not wrapper.connected:
                continue
            if wrapper.connected and not wrapper.has_telemetry:
                continue

            state = wrapper.current_telemetry
            speed = state.get("speed", -1)
            rpm = state.get("rpm", -1)
            gear = state.get("gear", -1)
            pos = state.get("position", {})
            x = pos.get("x", -1)
            y = pos.get("y", -1)
            z = pos.get("z", -1)
            

            print(f"Vitesse: {speed:.2f} | Pos: ({x:.2f}, {y:.2f}, {z:.2f}) | rpm: {rpm} | gear: {gear}")
    except KeyboardInterrupt:
        print("Arrêt du wrapper.")