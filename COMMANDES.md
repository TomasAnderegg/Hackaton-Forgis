# Commandes de lancement — Forgis Robot Control

## Prérequis
- Docker Desktop lancé
- Zivid Studio **fermé**
- Caméra Zivid branchée (IP : 192.168.15.107)
- Credentials Azure OpenAI renseignés dans `.env` (pour la génération de flows par LLM)

---

## 1. Serveur Zivid (terminal 1 — Windows natif)

```bash
cd assets/zivid-server
python zivid_server.py --port 8766 --fps 5
```

**Succès attendu :**
```
INFO - Zivid camera connected: Zivid 2+ MR130 (serial 2448G0B9)
INFO - Zivid WebSocket server starting on ws://0.0.0.0:8766
```

---

## 2. Docker (terminal 2)

```bash
docker compose up --build
```

> Enlever `--build` les prochaines fois si le code n'a pas changé :
> ```bash
> docker compose up
> ```

**Succès attendu dans les logs :**
```
[INFO] [zivid_node]: Connected to Zivid server
```

Vérifier les logs du backend :
```bash
docker logs forgis-backend --tail 30
```

---

## 3. Interface web

Ouvrir dans le navigateur : **http://localhost**

---

## Arrêt

```bash
# Arrêter Docker
docker compose down

# Arrêter le serveur Zivid : Ctrl+C dans le terminal 1
```

---

## Dépannage

### Caméra Zivid "busy"
→ Fermer Zivid Studio, puis relancer le serveur Zivid.

### Docker — conflit de containers
```bash
docker rm -f dobot-driver forgis-backend forgis-frontend
docker compose up
```

### Port 8766 bloqué (première fois)
Ouvrir PowerShell **en administrateur** :
```powershell
New-NetFirewallRule -DisplayName "Zivid WebSocket Server" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow
```

### Voir les logs en temps réel
```bash
docker logs forgis-backend -f
docker logs dobot-driver -f
```

---

## Génération de flow par LLM

Le endpoint `POST /api/flows/generate` accepte un prompt en langage naturel et génère un flow complet via Azure OpenAI (gpt-4o).

### Configuration (`.env`)
```
AZURE_OPENAI_ENDPOINT=https://<votre-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<votre-clé>
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Test rapide
```bash
curl -X POST http://localhost:8000/api/flows/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Pick a box from the conveyor and place it on the table using vision detection"}'
```

**Sans credentials** : retourne le flow par défaut `dobot_test_pick`.

### Démarrer le flow généré
```bash
# Remplacer <flow_id> par l'id retourné dans le champ "id"
curl -X POST http://localhost:8000/api/flows/<flow_id>/start
```
