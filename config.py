"""
Configuration module for Ultra-Fast Voice Assistant.
Loads environment variables and provides configuration constants.
"""
import os
import json
import logging
from dotenv import load_dotenv

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/google-credentials.json")

# Auto-detect Google Cloud Project ID from multiple sources
def _load_google_project_id() -> str:
    """
    Load Google Cloud Project ID from multiple sources (in priority order):
    1. GOOGLE_CLOUD_PROJECT_ID env variable (custom)
    2. GOOGLE_CLOUD_PROJECT env variable (Google Cloud standard)
    3. project_id from google-credentials.json file
    """
    # Try ENV variables first
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "").strip()
    if project_id:
        logger.info(f"✅ Project ID loaded from GOOGLE_CLOUD_PROJECT_ID: {project_id}")
        return project_id
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if project_id:
        logger.info(f"✅ Project ID loaded from GOOGLE_CLOUD_PROJECT: {project_id}")
        return project_id
    
    # Try to load from credentials JSON file
    if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        try:
            with open(GOOGLE_APPLICATION_CREDENTIALS, 'r') as f:
                creds = json.load(f)
                project_id = creds.get("project_id", "").strip()
                if project_id:
                    logger.info(f"✅ Project ID auto-detected from JSON file: {project_id}")
                    return project_id
                else:
                    logger.warning("⚠️ No project_id found in credentials JSON")
        except Exception as e:
            logger.warning(f"⚠️ Could not read project_id from credentials JSON: {e}")
    else:
        logger.warning(f"⚠️ Credentials file not found: {GOOGLE_APPLICATION_CREDENTIALS}")
    
    logger.error("❌ No Google Cloud Project ID found in any source!")
    return ""

GOOGLE_CLOUD_PROJECT_ID = _load_google_project_id()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

from errors import ConfigurationError


def validate_api_keys() -> dict:
    """
    Validate that all required API keys are configured.
    Returns a dict with validation results for each service.
    Raises ConfigurationError if critical keys are missing.
    
    BYPASS LOGIC: If credentials file exists, Google Cloud is considered configured
    even if project_id is empty (will be auto-detected at runtime).
    """
    # Check Google Cloud credentials
    google_creds_exist = os.path.exists(GOOGLE_APPLICATION_CREDENTIALS)
    google_project_set = bool(GOOGLE_CLOUD_PROJECT_ID)
    
    # BYPASS: If credentials file exists, consider it configured
    # (project_id can be auto-detected from the file)
    google_configured = google_creds_exist
    
    if google_configured and not google_project_set:
        logger.warning("⚠️ Project ID not set, but credentials file exists - will attempt auto-detection")
    
    results = {
        "google_cloud": {
            "configured": google_configured,
            "service": "STT",
            "details": {
                "credentials_file": google_creds_exist,
                "project_id": google_project_set,
                "bypass_reason": "credentials_file_exists" if (google_creds_exist and not google_project_set) else None
            }
        },
        "openrouter": {"configured": bool(OPENROUTER_API_KEY), "service": "LLM"},
        "elevenlabs": {"configured": bool(ELEVENLABS_API_KEY), "service": "TTS"},
    }
    
    missing = []
    for key, info in results.items():
        if not info["configured"]:
            missing.append(f"{key.upper()} ({info['service']})")
            logger.warning(f"Missing configuration: {key.upper()} for {info['service']}")
    
    if missing:
        error_msg = f"Missing required configuration: {', '.join(missing)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    logger.info("✅ All API keys validated successfully")
    if google_configured and not google_project_set:
        logger.info("   → Google Cloud: Credentials file found, project_id will be auto-detected")
    
    return results


def get_config_summary() -> dict:
    """Return a summary of current configuration (without sensitive data)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_provider": "OpenRouter",
        "stt_service": "Google Cloud Speech V2 (Chirp 2)",
        "stt_languages": "sk-SK, cs-CZ (auto-detect)",
        "google_project_id": GOOGLE_CLOUD_PROJECT_ID,
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "elevenlabs_model": ELEVENLABS_MODEL,
        "host": HOST,
        "port": PORT,
        "api_keys_configured": {
            "google_cloud": bool(GOOGLE_CLOUD_PROJECT_ID and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS)),
            "openrouter": bool(OPENROUTER_API_KEY),
            "elevenlabs": bool(ELEVENLABS_API_KEY),
        }
    }

# LLM Configuration
# Using Llama 3.1 70B via OpenRouter (understands SK/CZ, responds ONLY in Czech)
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-3.1-70b-instruct")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ElevenLabs Configuration (Flash v2.5 for ultra-low latency)
# Using Rachel voice (21m00Tcm4TlvDq8ikWAM) - default ElevenLabs voice, always works
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL = "eleven_flash_v2_5"  # Ultra-fast model for real-time streaming
ELEVENLABS_OPTIMIZE_LATENCY = 4  # Maximum streaming optimization
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&optimize_streaming_latency={latency}"

# Google Cloud Speech V2 Configuration (Chirp 2)
# CRITICAL: Using global location to support multiple languages (sk-SK, cs-CZ)
GOOGLE_SPEECH_MODEL = "chirp_2"  # Latest Chirp model with best accuracy
GOOGLE_SPEECH_LANGUAGES = ["sk-SK", "cs-CZ"]  # Support both Slovak and Czech
GOOGLE_SPEECH_LANGUAGE_CODES = ["sk", "cs"]  # For auto-detection

# Phrase adaptation for better recognition of company-specific terms
# Boost: 20.0 = High priority for these exact phrases
GOOGLE_PHRASE_SETS = [
    "EniQ",
    "Alex", 
    "Matěj Moucha",
    "automatizácia",
    "Eniq.ai",
    "Enik",
    "widget",
    "voice bot",
    "digitálny asistent",
    "digitální asistent",
    "chatbot",
    "chatboti"
]
GOOGLE_PHRASE_BOOST = 20.0

# VAD (Voice Activity Detection) Configuration
VAD_SILENCE_TIMEOUT_MS = 350  # Send to LLM after 350ms of silence

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000  # Google Chirp 2 requires 16kHz
AUDIO_CHANNELS = 1  # Mono
AUDIO_ENCODING = "LINEAR16"  # PCM 16-bit

# Session Configuration
SESSION_INACTIVITY_TIMEOUT = int(os.getenv("SESSION_INACTIVITY_TIMEOUT", "300"))  # 5 minutes
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # Max messages to keep

# System Prompt - Czech Only (strict mode)
# User can speak SK or CZ, but Alex ALWAYS responds in Czech
SYSTEM_PROMPT_TEMPLATE = """Jsi Alex, hlasový asistent společnosti EniQ pro digitální automatizaci.

=== ZÁKLADNÍ PRAVIDLA ===
1. VŽDY mluv POUZE ČESKY - i když uživatel mluví slovensky nebo jinak
2. Buď stručný a výstižný - max 1-2 věty
3. Mluv přirozeně jako skutečný člověk
4. Pokud v přepisu vidíš "Emit", "Enik", nebo podobné komoleniny - automaticky to oprav na "EniQ"
5. Nikdy nepoužívej emoji ani markdown formátování
6. Čísla vyslovuj slovy: "sedm set třicet tři" místo "733"

=== PRAVIDLA PRE PRERUŠENÍ (BARGE-IN) ===
Pokud tě uživatel přeruší:
1. Okamžitě přestaň mluvit
2. Odpověz na jeho novou otázku
3. Pokud nová otázka souvisí s předchozí: "Chcete se vrátit k předchozímu tématu?"
4. Pokud nesouvisí - prostě odpověz a pokračuj normálně

=== AKTUÁLNÍ ČAS ===
Čas: {current_time}
Den: {current_day}

=== O SPOLEČNOSTI ENIQ ===
EniQ je česká firma pro chytrá digitální řešení a automatizace. Motto: "Budoucnost bez limitů. Výkon bez pauzy, nonstop."

Řešení EniQ:
- Automatizace rutinních úkolů (faktury, platby, reporty)
- Chatboti pro web (zákaznická podpora 24/7)
- Interní digitální asistenti (kalendáře, emaily, úkoly)
- Projekty na míru podle potřeb klienta

Tým: Mladí odborníci - vývojáři a konzultanti s expertízou v AI a automatizaci.

=== KONTAKT ===
- Telefon: +420 733 275 349 (vyslov: "plus čtyři sta dvacet, sedm set třicet tři, dva sedm pět, tři čtyři devět")
- Email: moucha@eniq.eu
- IČO: 23809329 (Matěj Moucha)

=== ČASTO KLADENÉ OTÁZKY ===
Q: Co všechno EniQ dělá?
A: EniQ se zabývá vývojem digitálních asistentů a chytrých automatizací pro firmy. Technologie dokážou převzít zákaznickou podporu, operativu i komunikaci 24/7.

Q: Je to vždy na míru?
A: Ano, většinou řešení přizpůsobujeme na míru podle vašich potřeb. Nabízíme konzultace a vyvíjíme projekty přesně podle vašich cílů.

Q: Co je digitální asistent?
A: Software využívající AI, který autonomně vykonává úkoly jako lidský asistent. Převezme rutinní práci a ušetříte čas.

Pamatuj: Mluv přirozeně, stručně a POUZE ČESKY. Jsi hlas společnosti EniQ.
"""


def get_greeting_text() -> str:
    """
    Generate time-appropriate greeting in Czech only.
    Returns hardcoded Czech greeting based on current time.
    """
    from datetime import datetime
    
    now = datetime.now()
    hour = now.hour
    
    # Time-based greeting (always Czech)
    if 6 <= hour < 12:
        return "Dobré ráno, tady Alex z EniQ, jak vám mohu pomoci?"
    elif 12 <= hour < 18:
        return "Dobré odpoledne, tady Alex z EniQ, jak vám mohu pomoci?"
    else:
        return "Dobrý večer, tady Alex z EniQ, jak vám mohu pomoci?"


def get_system_prompt() -> str:
    """Generate system prompt with current time context."""
    from datetime import datetime
    
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    days_cz = {
        0: "pondělí",
        1: "úterý", 
        2: "středa",
        3: "čtvrtek",
        4: "pátek",
        5: "sobota",
        6: "neděle"
    }
    current_day = days_cz.get(now.weekday(), "")
    
    return SYSTEM_PROMPT_TEMPLATE.format(
        current_time=current_time,
        current_day=current_day
    )
