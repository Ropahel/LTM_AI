Net::Socket@ g_Socket = null;
bool g_Connected = false;

void Main() {
    print("[LTM Telemetry] Plugin chargé.");
    while (true) {
        yield();
        EnsureConnected();
        SendTelemetrySafe();
    }
}

void EnsureConnected() {
    if (g_Connected && g_Socket !is null) return;

    if (g_Socket is null) {
        @g_Socket = Net::Socket();
    }

    bool ok = g_Socket.Connect("127.0.0.1", 9000);
    if (ok) {
        g_Connected = true;
    } else {
        g_Connected = false;
    }
}

void SendTelemetrySafe() {
    if (!g_Connected || g_Socket is null) return;

    bool ok = SendTelemetry();
    if (!ok) {
        g_Connected = false;
    }
}

bool SendTelemetry() {
    // Récupération de la voiture "vue" par le joueur
    CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
    if (vis is null) {
        // Menu, chargement, etc. -> pas d’erreur réseau, juste rien à envoyer
        return true;
    }

    // Vecteur vitesse monde
    vec3 vel = vis.WorldVel;

    // Vitesse scalaire en m/s
    float speed_ms = vel.Length();

    // RPM et Gear
    float rpm = VehicleState::GetRPM(vis);
    int gear = int(vis.CurGear);

    Json::Value data = Json::Object();
    data["speed"] = speed_ms;
    data["rpm"] = rpm;
    data["gear"] = gear;

    // Position pour debug / futur usage
    vec3 pos = vis.Position;
    Json::Value posJson = Json::Object();
    posJson["x"] = pos.x;
    posJson["y"] = pos.y;
    posJson["z"] = pos.z;
    data["position"] = posJson;

    string payload = Json::Write(data) + "\n";
    bool ok = g_Socket.Write(payload);
    return ok;
}