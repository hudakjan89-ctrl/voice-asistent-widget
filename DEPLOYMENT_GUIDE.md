# 🚀 Ultra-Fast Voice AI - Deployment Guide

## ⚡ Očakávaná výkonnosť:
- **End-to-end latency**: <1.5s
- **STT (Google Chirp 2)**: ~200-300ms
- **LLM (Llama 3.3 70B)**: ~400-600ms
- **TTS (ElevenLabs Flash v2.5)**: ~200-400ms

---

## 📋 Pred-deploy checklist:

### 1️⃣ Google Cloud Setup (KRITICKÉ!)

1. **Vytvor Google Cloud projekt:**
   ```
   → https://console.cloud.google.com/
   → Create New Project
   → Poznač si Project ID (napr. "voice-ai-123456")
   ```

2. **Povoľ Speech-to-Text API v2:**
   ```
   → Navigation Menu → APIs & Services → Library
   → Hľadaj: "Cloud Speech-to-Text API"
   → Enable
   ```

3. **Vytvor Service Account:**
   ```
   → Navigation Menu → IAM & Admin → Service Accounts
   → Create Service Account
   → Name: "voice-ai-service"
   → Role: "Cloud Speech Client"
   → Create and continue → Done
   ```

4. **Stiahni JSON credentials:**
   ```
   → Klikni na vytvorený Service Account
   → Keys tab → Add Key → Create new key → JSON
   → Stiahne sa súbor (napr. "voice-ai-123456-abc123.json")
   → ULOŽ TENTO SÚBOR! Budeš ho potrebovať pre Coolify
   ```

---

### 2️⃣ Coolify Deployment

#### A) Vytvor aplikáciu v Coolify:
```
Type: Git Repository
Repository: https://github.com/hudakjan89-ctrl/voice-asistent-widget
Branch: main
Build Pack: Python
```

#### B) Nahraj Google credentials:

**METÓDA: Storage → Files**

1. V Coolify: Choď do **Storage** → **Files**
2. Klikni **Create File**
3. Nastavenia:
   ```
   Path: /app/google-credentials.json
   Content: [Vlož celý obsah stiahnutého JSON súboru]
   ```
4. **Save**

#### C) Nastav Environment Variables:

Choď do **Environment** tab a pridaj:

```bash
# ========== REQUIRED ==========

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json

# Google Cloud Project ID (use either of these - both work, will auto-detect from JSON if missing)
GOOGLE_CLOUD_PROJECT=tvoj-project-id-z-kroku-1
# OR use this (both work):
# GOOGLE_CLOUD_PROJECT_ID=tvoj-project-id-z-kroku-1

# OpenRouter (už máš)
OPENROUTER_API_KEY=sk_or_v1_xxxxx

# ElevenLabs (už máš)
ELEVENLABS_API_KEY=sk_xxxxx

# ========== OPTIONAL ==========

# LLM Model (default: Llama 3.1 70B via OpenRouter)
LLM_MODEL=meta-llama/llama-3.1-70b-instruct

# Voice (default: Rachel)
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# Server
HOST=0.0.0.0
PORT=8000

# Session
SESSION_INACTIVITY_TIMEOUT=300
MAX_CONVERSATION_HISTORY=20
```

#### D) Deploy:
1. Klikni **Deploy**
2. Sleduj logs v **Deployment Logs**
3. Počkaj na "Application startup complete"

---

## 🧪 Testing (po deployi):

### 1. Health Check:
```bash
curl https://tvoja-domena.duckdns.org/health
```

**Očakávaný output:**
```json
{
  "status": "healthy",
  "service": "ultra-fast-voice-assistant",
  "config": {
    "llm_model": "meta-llama/llama-3.1-70b-instruct",
    "llm_provider": "OpenRouter",
    "stt_service": "Google Cloud Speech V2 (Chirp 2)",
    "stt_languages": "sk-SK, cs-CZ (auto-detect)",
    "google_project_id": "your-project-id",
    "api_keys_configured": {
      "google_cloud": true,
      "openrouter": true,
      "elevenlabs": true
    }
  }
}
```

### 2. Detailed Health Check:
```bash
curl https://tvoja-domena.duckdns.org/health/detailed
```

**Skontroluj:**
- `google_cloud.status`: "ok"
- `openrouter.status`: "ok"
- `elevenlabs.status`: "ok"

### 3. Frontend Test:
```
→ Otvor: https://tvoja-domena.duckdns.org/
→ Klikni na modrú ikonu mikrofónu
→ Počuj greeting: "Dobré odpoledne, tady Alex z EniQ..."
→ Povedz: "Čo robí EniQ?" (slovensky alebo česky)
→ Asistent odpovie ČESKY do 1.5s
```

---

## 🐛 Troubleshooting:

### ❌ "Missing required configuration: GOOGLE_CLOUD"
**Riešenie:**
1. Skontroluj, že `/app/google-credentials.json` existuje v Storage → Files
2. Skontroluj, že `GOOGLE_APPLICATION_CREDENTIALS=/app/google-credentials.json` je v ENV
3. Skontroluj, že `GOOGLE_CLOUD_PROJECT_ID` je správny (z Google Console)

### ❌ "Failed to initialize Google Speech V2"
**Riešenie:**
1. Skontroluj, že Speech-to-Text API je **Enabled** v Google Console
2. Skontroluj, že Service Account má rolu **"Cloud Speech Client"**
3. Skontroluj logs: `curl https://tvoja-domena.duckdns.org/health/detailed`

### ❌ "No audio output" / "Voice ID does not exist"
**Riešenie:**
1. Skontroluj ElevenLabs API key: https://elevenlabs.io/
2. Skúsi iný voice:
   ```bash
   ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel
   ```
3. Redeploy

### ❌ "STT not recognizing Slovak/Czech"
**Riešenie:**
1. Google Chirp 2 má phrase adaptation pre: `EniQ`, `Alex`, `Matěj Moucha`, atď.
2. Ak to stále zle rozpoznáva, môžeš pridať vlastné phrases do `config.py`:
   ```python
   GOOGLE_PHRASE_SETS = [
       "EniQ",
       "tvoje-vlastne-slovo",
       # ...
   ]
   ```

### ❌ "Latency >2s"
**Riešenie:**
1. Skontroluj region: Google Cloud by mal byť v EU (europe-west1)
2. Skontroluj OpenRouter logs - možno je preťažený
3. ElevenLabs Flash v2.5 má `optimize_streaming_latency=4` (max optimization)

---

## 📊 Performance Monitoring:

### Backend Logs (Coolify):
```
✅ "Google Speech V2 (Chirp 2) initialized"
✅ "Connected to ElevenLabs Flash v2.5"
✅ "🎯 Final transcript (cs): ..."
✅ "🧠 LLM generating response for: ..."
✅ "✅ LLM response complete: X chars"
```

### Frontend Console (DevTools):
```
✅ "Microphone access granted"
✅ "WebSocket connected"
✅ "Audio processing started"
```

---

## 🔐 Security Notes:

1. **NIKDY** necommituj `google-credentials.json` do Gitu
2. **NIKDY** nezdieľaj `GOOGLE_CLOUD_PROJECT_ID` verejne
3. V Google Console: Restrikuj API key na konkrétnu IP (optional)

---

## 🎉 Next Steps:

1. **Otestuj rôzne scenáre:**
   - Slovenský input → Český output ✅
   - Český input → Český output ✅
   - Barge-in (prerušenie bota) ✅
   - Phrase adaptation ("EniQ" namiesto "Emit") ✅

2. **Monitoring:**
   - Sleduj Google Cloud Speech usage: https://console.cloud.google.com/apis/api/speech.googleapis.com/metrics
   - Free tier: 60 minutes/month
   - Potom: $0.006/15s audio

3. **Optimalizácia (ak je potrebná):**
   - Znížiť `max_tokens` v LLM (config.py)
   - Zmeniť `VAD_SILENCE_TIMEOUT_MS` (config.py)
   - Upraviť `GOOGLE_PHRASE_BOOST` (config.py)

---

**Enjoy your ultra-fast voice AI! ⚡🎤**
