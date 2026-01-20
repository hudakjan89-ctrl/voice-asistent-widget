# Ultra-Low Latency Voice Assistant Backend

Hlasový AI asistent s odozvou pod 1 sekundu. Backend pre real-time konverzáciu pomocou WebSocket streamingu.

## Technologický Stack

- **Server:** FastAPI + Uvicorn (asynchrónny)
- **Protokol:** WebSockets (obojsmerný stream audia)
- **STT (Uši):** Deepgram API (nova-2, streaming)
- **LLM (Mozog):** OpenRouter API (kompatibilné s OpenAI)
- **TTS (Ústa):** ElevenLabs API (eleven_turbo_v2_5, streaming)
- **Kontajnerizácia:** Docker + Docker Compose

## Funkcie

- ✅ **Instantný pozdrav** - Bot sa ozve ihneď po pripojení
- ✅ **Real-time streaming** - Dáta tečú prúdom bez čakania na celú vetu
- ✅ **Barge-in** - Prerušenie bota počas rozprávania
- ✅ **Text normalizácia** - Čísla a skratky sú prevedené na hovorenú formu
- ✅ **Český/Slovenský vstup** - Rozpoznávanie CZ/SK jazyka
- ✅ **Český výstup** - Bot odpovedá v češtine

## Rýchly štart

### 1. Konfigurácia

```bash
cd app
cp .env.example .env
```

Upravte `.env` súbor a doplňte svoje API kľúče:

```env
DEEPGRAM_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

### 2. Spustenie s Dockerom

```bash
docker-compose up --build
```

### 3. Spustenie bez Dockera (vývoj)

```bash
pip install -r requirements.txt
python -m uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Testovanie

Otvorte prehliadač na `http://localhost:8000`

## API Endpoints

### WebSocket `/ws/audio`

Hlavný endpoint pre audio streaming.

**Klientský protokol:**
- Posielať: binárne audio dáta (PCM 16-bit, 16kHz, mono)
- Prijímať: binárne audio dáta (MP3) + JSON správy

**JSON správy od servera:**

```json
{"type": "session_started"}
{"type": "user_text", "text": "...", "is_final": true/false}
{"type": "assistant_text", "text": "...", "is_final": true/false}
{"type": "clear_audio"}  // Barge-in - vymazať audio buffer
{"type": "audio_end"}
{"type": "session_timeout", "message": "..."}  // Ukončenie kvôli neaktivite
{"type": "error", "code": "...", "message": "..."}
```

**JSON správy pre server:**

```json
{"type": "ping"}
{"type": "stop"}
```

### HTTP Endpoints

- `GET /` - Testovací klient
- `GET /health` - Základný health check
- `GET /health/detailed` - Detailný health check s testovaním externých služieb

### Health Check Response

**Základný (`/health`):**
```json
{
  "status": "healthy",
  "service": "voice-assistant",
  "config": {
    "llm_model": "anthropic/claude-3.5-haiku",
    "deepgram_model": "nova-2-general",
    "deepgram_language": "cs",
    "api_keys_configured": {
      "deepgram": true,
      "openrouter": true,
      "elevenlabs": true
    }
  }
}
```

**Detailný (`/health/detailed`):**
```json
{
  "status": "healthy",
  "service": "voice-assistant",
  "checks": {
    "deepgram": {"status": "ok", "reachable": true, "authenticated": true},
    "openrouter": {"status": "ok", "reachable": true, "authenticated": true},
    "elevenlabs": {"status": "ok", "reachable": true, "authenticated": true}
  }
}
```

## Architektúra

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│   Klient    │◄───►│              FastAPI Server                  │
│  (Browser)  │     │                                              │
│             │     │  ┌─────────┐  ┌─────────┐  ┌──────────────┐ │
│  Mikrofón ──┼────►│  │Deepgram │─►│   LLM   │─►│  ElevenLabs  │ │
│             │     │  │  (STT)  │  │(OpenAI) │  │    (TTS)     │ │
│  Speaker ◄──┼─────│  └─────────┘  └─────────┘  └──────────────┘ │
│             │     │         ▲                          │         │
│             │     │         └── Barge-in Detection ────┘         │
└─────────────┘     └──────────────────────────────────────────────┘
```

## Konfigurácia LLM Modelu

V `.env` nastavte model podľa potreby:

```env
# Pre prirodzenejší dialóg (pomalšie)
LLM_MODEL=nousresearch/hermes-3-llama-3.1-70b

# Pre maximálnu rýchlosť
LLM_MODEL=anthropic/claude-3.5-haiku
```

## Úprava System Promptu

System prompt sa nachádza v `server/config.py` v premennej `SYSTEM_PROMPT_TEMPLATE`.
Obsahuje placeholder pre kontext firmy (až A4 textu).

## Limity a výkon

- **RAM limit:** 2GB (nastavené v docker-compose.yml)
- **Odporúčaná odozva:** < 1 sekunda
- **Audio formát:** PCM 16-bit, 16kHz, mono (vstup)
- **Audio formát:** MP3 (výstup od ElevenLabs)
- **Timeout neaktivity:** 5 minút (konfigurovateľné cez `SESSION_INACTIVITY_TIMEOUT`)
- **História konverzácie:** max 20 správ (konfigurovateľné cez `MAX_CONVERSATION_HISTORY`)

## Funkcie reliability

- **Auto-reconnect:** Automatické znovupripojenie k Deepgram pri výpadku (max 3 pokusy)
- **Session timeout:** Automatické ukončenie neaktívnych relácií
- **API validácia:** Kontrola API kľúčov pri štarte servera
- **Health checks:** Detailné health check endpointy pre monitoring

## Riešenie problémov

### Vysoká latencia
- Skontrolujte internetové pripojenie
- Vyskúšajte rýchlejší LLM model (claude-3.5-haiku)
- Skontrolujte logy: `docker-compose logs -f`

### Problémy s mikrofónom
- Uistite sa, že prehliadač má povolený prístup k mikrofónu
- Používajte HTTPS alebo localhost

### WebSocket sa nepripojí
- Skontrolujte firewall nastavenia
- Overte, že server beží: `curl http://localhost:8000/health`

## Licencia

MIT
