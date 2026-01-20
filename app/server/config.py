"""
Configuration module for Voice Assistant.
Loads environment variables and provides configuration constants.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ElevenLabs Configuration
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = "eleven_turbo_v2_5"
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

# Deepgram Configuration
DEEPGRAM_MODEL = "nova-2-general"
DEEPGRAM_LANGUAGE = "cs"  # Czech language

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# System Prompt Template (Placeholder for A4 text)
SYSTEM_PROMPT_TEMPLATE = """
Jsi profesionální hlasový asistent pro zákaznickou podporu. Tvoje jméno je Alex.

PRAVIDLA KOMUNIKACE:
1. Vždy odpovídej POUZE v češtině, bez ohledu na to, jakým jazykem zákazník mluví.
2. Buď stručný a věcný - krátké odpovědi jsou lepší.
3. Buď profesionální, ale přátelský.
4. Pokud něčemu nerozumíš, zdvořile požádej o upřesnění.
5. Nikdy nepoužívej emoji ani speciální znaky.

AKTUÁLNÍ ČAS: {current_time}
AKTUÁLNÍ DEN: {current_day}

KONTEXT SPOLEČNOSTI:
[PLACEHOLDER - Zde bude vložen kontext společnosti, produkty, služby, FAQ atd. 
Tento text může mít rozsah až A4 strany a bude obsahovat všechny relevantní informace,
které asistent potřebuje znát pro poskytování kvalitní zákaznické podpory.]

PŘÍKLADY ODPOVĚDÍ:
- Pozdrav ráno (6:00-12:00): "Dobré ráno, tady Alex, jak vám mohu pomoci?"
- Pozdrav odpoledne (12:00-18:00): "Dobré odpoledne, tady Alex, jak vám mohu pomoci?"
- Pozdrav večer (18:00-22:00): "Dobrý večer, tady Alex, jak vám mohu pomoci?"
- Pozdrav noc (22:00-6:00): "Dobrý večer, tady Alex, jak vám mohu pomoci?"

Pamatuj: Odpovídej přirozeně a konverzačně, jako skutečný člověk.
"""


def get_greeting_prompt() -> str:
    """Generate a greeting prompt based on current time."""
    from datetime import datetime
    import locale
    
    try:
        locale.setlocale(locale.LC_TIME, 'cs_CZ.UTF-8')
    except:
        pass
    
    now = datetime.now()
    hour = now.hour
    
    if 6 <= hour < 12:
        time_of_day = "ráno"
        greeting = "Dobré ráno"
    elif 12 <= hour < 18:
        time_of_day = "odpoledne"
        greeting = "Dobré odpoledne"
    elif 18 <= hour < 22:
        time_of_day = "večer"
        greeting = "Dobrý večer"
    else:
        time_of_day = "noc"
        greeting = "Dobrý večer"
    
    return f"Vygeneruj krátký pozdrav pro zákazníka. Je {time_of_day}. Použij pozdrav '{greeting}' a představ se jako Alex. Maximálně jedna věta."


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
