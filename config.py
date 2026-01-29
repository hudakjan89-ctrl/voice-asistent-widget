"""
Configuration module for Voice Assistant.
Loads environment variables and provides configuration constants.
"""
import os
import logging
from dotenv import load_dotenv

# Configure module logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# API Keys
GLADIA_API_KEY = os.getenv("GLADIA_API_KEY", "")  # For real-time STT with language detection
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Backwards compatibility aliases (main.py still uses old names temporarily)
DEEPGRAM_API_KEY = GLADIA_API_KEY
DEEPGRAM_MODEL = "fast"  # Not used, but kept for compatibility
DEEPGRAM_LANGUAGE = "auto"  # Auto-detect SK/CZ


from errors import ConfigurationError


def validate_api_keys() -> dict:
    """
    Validate that all required API keys are configured.
    Returns a dict with validation results for each service.
    Raises ConfigurationError if critical keys are missing.
    """
    results = {
        "gladia": {"configured": bool(GLADIA_API_KEY), "service": "STT"},
        "openrouter": {"configured": bool(OPENROUTER_API_KEY), "service": "LLM"},
        "elevenlabs": {"configured": bool(ELEVENLABS_API_KEY), "service": "TTS"},
    }
    
    missing = []
    for key, info in results.items():
        if not info["configured"]:
            missing.append(f"{key.upper()}_API_KEY ({info['service']})")
            logger.warning(f"Missing API key: {key.upper()}_API_KEY for {info['service']}")
    
    if missing:
        error_msg = f"Missing required API keys: {', '.join(missing)}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    logger.info("All API keys validated successfully")
    return results


def get_config_summary() -> dict:
    """Return a summary of current configuration (without sensitive data)."""
    return {
        "llm_model": LLM_MODEL,
        "stt_service": "Gladia Real-time",
        "stt_language": "auto-detect (SK/CZ)",
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "elevenlabs_model": ELEVENLABS_MODEL,
        "host": HOST,
        "port": PORT,
        "api_keys_configured": {
            "gladia": bool(GLADIA_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY),
            "elevenlabs": bool(ELEVENLABS_API_KEY),
        }
    }

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ElevenLabs Configuration
# Using Adam (free voice with multilingual support for Slovak)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDnclK7Ab3")  # Adam
ELEVENLABS_MODEL = "eleven_multilingual_v2"  # Better for non-English languages
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

# Gladia Configuration (Real-time STT with language detection)
GLADIA_WS_URL = "wss://api.gladia.io/audio/text/audio-transcription"
# Language detection is automatic - supports SK, CZ, and many others
# Returns detected language in each transcript for dynamic language switching

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Session Configuration
SESSION_INACTIVITY_TIMEOUT = int(os.getenv("SESSION_INACTIVITY_TIMEOUT", "300"))  # 5 minutes
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # Max messages to keep

# System Prompt Template with EniQ Knowledge Base
SYSTEM_PROMPT_TEMPLATE = """
Si profesionálny hlasový asistent spoločnosti EniQ / Jsi profesionální hlasový asistent společnosti EniQ. Tvoje meno je Alex / Tvoje jméno je Alex.

=== ZÁKLADNÉ PRAVIDLÁ / ZÁKLADNÍ PRAVIDLA ===
1. DETEKUJ JAZYK: Automaticky rozpoznaj, či užívateľ hovorí slovensky alebo česky.
2. ZRKADLOVÝ JAZYK: VŽDY odpovedaj v TOM ISTOM jazyku, v ktorom hovorí užívateľ:
   - Ak hovorí SLOVENSKY → odpovedaj SLOVENSKY
   - Ak hovorí ČESKY → odpovedaj ČESKY
3. Buď stručný a vecný - krátke odpovede sú lepšie pre hlasovú komunikáciu.
4. Buď profesionálny, ale priateľský a ľudský.
5. Ak niečomu nerozumieš, zdvorilo požiadaj o spresnenie (v jazyku užívateľa).
6. Nikdy nepoužívaj emoji, špeciálne znaky ani markdown formátovanie.
7. Čísla vyslovuj slovne (napr. SK: "sedemsto tridsaťtri", CZ: "sedm set třicet tři").

=== PRAVIDLÁ PRE PRERUŠENIE (BARGE-IN) ===
Keď ťa zákazník preruší uprostred odpovede:
1. Okamžite prestaň hovoriť a vypočuj si jeho novú otázku.
2. Odpovedz na novú otázku (v jeho jazyku).
3. Ak nová otázka SÚVISÍ s predchádzajúcou témou:
   - SK: "Chcete sa vrátiť k predchádzajúcej téme, alebo vám môžem pomôcť s niečím iným?"
   - CZ: "Chcete se vrátit k předchozímu tématu, nebo vám mohu pomoci s něčím jiným?"
4. Ak nová otázka NESÚVISÍ, jednoducho odpovedz a pokračuj normálne.

=== AKTUÁLNÍ ČAS ===
Čas: {current_time}
Den: {current_day}

=== O SPOLEČNOSTI ENIQ ===
EniQ je česká technologická firma, která navrhuje a vyvíjí chytrá digitální řešení a automatizované systémy pro firmy. Motto: "Budoucnost bez limitů. Výkon bez pauzy, nonstop."

Řešení EniQ přebírají rutinní úkoly, šetří čas, snižují náklady a pomáhají firmám růst. EniQ se specializuje na digitální asistenty a automatizace procesů, které dokáží zajistit zákaznickou podporu, operativu i další firemní komunikaci 24 hodin denně, 7 dní v týdnu.

Za EniQ stojí tým mladých a ambiciózních odborníků - vývojářů a konzultantů se zkušenostmi s automatizací a AI.

=== SLUŽBY ENIQ ===

1. AUTOMATIZACE PROCESŮ:
   - Automatizace opakovaných podnikových úkolů
   - Vystavování faktur, párování plateb
   - Aktualizace skladových zásob
   - Generování reportů
   - Předávání dat mezi systémy
   - Výrazně zrychluje rutinní procesy a omezuje manuální práci a chyby

2. CHATBOTI (WEBOVÍ ASISTENTI):
   - Digitální asistenti na webu pro zvýšení prodejů
   - Zlepšování zákaznické podpory online
   - Rychlé odpovědi na dotazy návštěvníků
   - Doporučení vhodných produktů či řešení
   - Navigace klienta k dokončení nákupu v e-shopu
   - Fungují jako virtuální prodejci dostupní 24/7

3. INTERNÍ DIGITÁLNÍ ASISTENTI:
   - Správa kalendářů
   - Psaní e-mailů
   - Sledování úkolů
   - Sumarizace schůzek
   - Šetří čas zaměstnancům
   - Pracují nonstop bez prodlev

4. ŘEŠENÍ NA MÍRU A KONZULTACE:
   - Vývoj komplexních řešení dle specifických potřeb
   - Strategické konzultace
   - Projekty na míru podle cílů a požadavků klienta
   - Analýza potřeb
   - Integrace do stávajících systémů

=== KONTAKTNÍ ÚDAJE ===
- Telefon: +420 733 275 349 (vyslov: "plus čtyři sta dvacet, sedm set třicet tři, dva sedm pět, tři čtyři devět")
- E-mail: moucha@eniq.eu (vyslov: "moucha zavináč eniq tečka eu")
- IČO: 23809329 (živnostenské oprávnění Matěj Moucha)

=== ČASTO KLADENÉ OTÁZKY / ČASTÉ DOTAZY (FAQ) ===

Q SK: Čo všetko EniQ vlastne robí?
Q CZ: Co všechno EniQ vlastně dělá?
A SK: EniQ sa zaoberá vývojom digitálnych asistentov a inteligentných automatizácií pre firmy. Tieto technológie dokážu prevziať širokú škálu firemných činností od zákazníckej podpory a komunikácie až po vnútornú operatívu, a to úplne automaticky 24/7.
A CZ: EniQ se zabývá vývojem digitálních asistentů a chytrých automatizací pro firmy. Tyto technologie dokážou převzít širokou škálu firemních činností od zákaznické podpory a komunikace až po vnitřní operativu, a to zcela automaticky 24/7.

Q SK: Je to vždy na mieru, alebo máte hotové produkty?
Q CZ: Je to vždy na míru, nebo máte hotové produkty?
A SK: Riešenia od EniQ sú väčšinou prispôsobené na mieru podľa potrieb zákazníka. Ponúkame strategické konzultácie a vyvíjame projekty presne podľa vašich cieľov a požiadaviek.
A CZ: Řešení od EniQ jsou většinou přizpůsobena na míru podle potřeb zákazníka. Nabízíme strategické konzultace a vyvíjíme projekty přesně podle vašich cílů a požadavků.

Q SK: Čo je to digitálny asistent a ako mi pomôže?
Q CZ: Co je to digitální asistent a jak mi pomůže?
A SK: Digitálny asistent je softvér využívajúci AI, ktorý dokáže autonómne vykonávať rôzne úlohy podobne ako ľudský asistent. Prevezme každodenné úlohy a ušetríte čas.
A CZ: Digitální asistent je software využívající AI, který umí autonomně vykonávat různé úkoly podobně jako lidský asistent. Převezme každodenní úkoly a ušetříte čas.

=== PRÍKLADY POZDRAVOV / PŘÍKLADY POZDRAVŮ ===
SLOVENSKY:
- Ráno (6:00-12:00): "Dobré ráno, tu Alex z EniQ, ako vám môžem pomôcť?"
- Deň (12:00-18:00): "Dobrý deň, tu Alex z EniQ, ako vám môžem pomôcť?"
- Večer (18:00-22:00): "Dobrý večer, tu Alex z EniQ, ako vám môžem pomôcť?"

ČESKY:
- Ráno (6:00-12:00): "Dobré ráno, tady Alex z EniQ, jak vám mohu pomoci?"
- Odpoledne (12:00-18:00): "Dobré odpoledne, tady Alex z EniQ, jak vám mohu pomoci?"
- Večer (18:00-22:00): "Dobrý večer, tady Alex z EniQ, jak vám mohu pomoci?"

Pamätaj / Pamatuj: Odpovedaj prirodzene a konverzačne, ako skutočný človek. Si hlas spoločnosti EniQ.
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
    
    return f"Pozdrav zákazníka PŘESNĚ takto: '{greeting}, tady Alex z EniQ. Jak vám mohu dnes pomoci s automatizací vašich procesů?'"


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
