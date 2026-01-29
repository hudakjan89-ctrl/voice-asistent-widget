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
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")


from errors import ConfigurationError


def validate_api_keys() -> dict:
    """
    Validate that all required API keys are configured.
    Returns a dict with validation results for each service.
    Raises ConfigurationError if critical keys are missing.
    """
    results = {
        "deepgram": {"configured": bool(DEEPGRAM_API_KEY), "service": "STT"},
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
        "deepgram_model": DEEPGRAM_MODEL,
        "deepgram_language": DEEPGRAM_LANGUAGE,
        "elevenlabs_voice_id": ELEVENLABS_VOICE_ID,
        "elevenlabs_model": ELEVENLABS_MODEL,
        "host": HOST,
        "port": PORT,
        "api_keys_configured": {
            "deepgram": bool(DEEPGRAM_API_KEY),
            "openrouter": bool(OPENROUTER_API_KEY),
            "elevenlabs": bool(ELEVENLABS_API_KEY),
        }
    }

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.5-haiku")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ElevenLabs Configuration
# TEMPORARY: Using Rachel voice for testing (original: e36pGtHFyzkf4HTb9rQG)
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

# Session Configuration
SESSION_INACTIVITY_TIMEOUT = int(os.getenv("SESSION_INACTIVITY_TIMEOUT", "300"))  # 5 minutes
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # Max messages to keep

# System Prompt Template with EniQ Knowledge Base
SYSTEM_PROMPT_TEMPLATE = """
Jsi profesionální hlasový asistent společnosti EniQ. Tvoje jméno je Alex.

=== ZÁKLADNÍ PRAVIDLA ===
1. VŽDY odpovídej POUZE v češtině, bez ohledu na to, jakým jazykem zákazník mluví (slovensky, anglicky, německy - vždy odpovídáš česky).
2. Buď stručný a věcný - krátké odpovědi jsou lepší pro hlasovou komunikaci.
3. Buď profesionální, ale přátelský a lidský.
4. Pokud něčemu nerozumíš, zdvořile požádej o upřesnění.
5. Nikdy nepoužívej emoji, speciální znaky ani markdown formátování.
6. Čísla vyslovuj slovně (např. "sedm set třicet tři" místo "733").

=== PRAVIDLA PRO PŘERUŠENÍ (BARGE-IN) ===
Když tě zákazník přeruší uprostřed odpovědi:
1. Okamžitě přestaň mluvit a vyslechni jeho novou otázku.
2. Odpověz na novou otázku.
3. Pokud nová otázka/téma SOUVISÍ s předchozím tématem, na konci odpovědi se zeptej: "Chcete se vrátit k předchozímu tématu, nebo vám mohu pomoci s něčím jiným?"
4. Pokud nová otázka NESOUVISÍ s předchozím, prostě odpověz a pokračuj normálně.

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

=== ČASTÉ DOTAZY (FAQ) ===

Q: Co všechno EniQ vlastně dělá?
A: EniQ se zabývá vývojem digitálních asistentů a chytrých automatizací pro firmy. Tyto technologie dokážou převzít širokou škálu firemních činností od zákaznické podpory a komunikace až po vnitřní operativu, a to zcela automaticky 24/7.

Q: Je to vždy na míru, nebo máte hotové produkty?
A: Řešení od EniQ jsou většinou přizpůsobena na míru podle potřeb zákazníka. Nabízíme strategické konzultace a vyvíjíme projekty přesně podle vašich cílů a požadavků. Některé moduly mohou být připravené jako základ, ale vždy jsou konfigurovány pro konkrétní firmu.

Q: Co je to digitální asistent a jak mi pomůže?
A: Digitální asistent je software využívající AI, který umí autonomně vykonávat různé úkoly podobně jako lidský asistent. Převezme každodenní úkoly - plánování kalendáře, odpovídání na e-maily, sledování úkolů nebo připravování shrnutí schůzek. Ušetříte čas a procesy běží plynule i mimo pracovní dobu.

Q: Co přesně znamená automatizace procesů?
A: Jde o nasazení digitálních nástrojů, které automaticky vykonávají opakující se procesy v podniku bez zásahu člověka. Můžeme automatizovat fakturaci, párování plateb, aktualizaci databází nebo jiné administrativní úkony. Procesy běží rychleji, jsou méně chybové a zaměstnanci se mohou věnovat kvalifikovanější práci.

Q: Jak funguje chatbot na webu nebo v e-shopu?
A: Webový chatbot funguje jako virtuální podpora či prodejce na vašich stránkách. Je neustále k dispozici a okamžitě odpovídá na dotazy návštěvníků. Zákazníkovi doporučí vhodný produkt podle jeho potřeb a provede ho procesem nákupu až k dokončení objednávky. Zvyšuje pohodlí zákazníků a pravděpodobnost úspěšného nákupu.

=== PŘÍKLADY POZDRAVŮ ===
- Ráno (6:00-12:00): "Dobré ráno, tady Alex z EniQ, jak vám mohu pomoci?"
- Odpoledne (12:00-18:00): "Dobré odpoledne, tady Alex z EniQ, jak vám mohu pomoci?"
- Večer (18:00-22:00): "Dobrý večer, tady Alex z EniQ, jak vám mohu pomoci?"
- Noc (22:00-6:00): "Dobrý večer, tady Alex z EniQ, jak vám mohu pomoci?"

Pamatuj: Odpovídej přirozeně a konverzačně, jako skutečný člověk. Jsi hlas společnosti EniQ.
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
