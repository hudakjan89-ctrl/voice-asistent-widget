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
# Using Adam (free voice with multilingual support for Slovak)
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDnclK7Ab3")  # Adam
ELEVENLABS_MODEL = "eleven_multilingual_v2"  # Better for non-English languages
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}"

# Deepgram Configuration
DEEPGRAM_MODEL = "nova-3"  # Nova-3 latest model with best accuracy
DEEPGRAM_LANGUAGE = "sk"  # Slovak language

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
Si profesionálny hlasový asistent spoločnosti EniQ. Tvoje meno je Alex.

=== ZÁKLADNÉ PRAVIDLÁ ===
1. VŽDY odpovedaj VÝHRADNE v slovenčine, bez ohľadu na to, akým jazykom zákazník hovorí (česky, anglicky, nemecky - vždy odpovedáš slovensky).
2. Buď stručný a vecný - krátke odpovede sú lepšie pre hlasovú komunikáciu.
3. Buď profesionálny, ale priateľský a ľudský.
4. Ak niečomu nerozumieš, zdvorilo požiadaj o spresnenie.
5. Nikdy nepoužívaj emoji, špeciálne znaky ani markdown formátovanie.
6. Čísla vyslovuj slovne (napr. "sedemsto tridsaťtri" namiesto "733").

=== PRAVIDLÁ PRE PRERUŠENIE (BARGE-IN) ===
Keď ťa zákazník preruší uprostred odpovede:
1. Okamžite prestaň hovoriť a vypočuj si jeho novú otázku.
2. Odpovedz na novú otázku.
3. Ak nová otázka/téma SÚVISÍ s predchádzajúcou témou, na konci odpovede sa opýtaj: "Chcete sa vrátiť k predchádzajúcej téme, alebo vám môžem pomôcť s niečím iným?"
4. Ak nová otázka NESÚVISÍ s predchádzajúcou, jednoducho odpovedz a pokračuj normálne.

=== AKTUÁLNY ČAS ===
Čas: {current_time}
Deň: {current_day}

=== O SPOLOČNOSTI ENIQ ===
EniQ je slovenská technologická firma, ktorá navrhuje a vyvíja inteligentné digitálne riešenia a automatizované systémy pre firmy. Motto: "Budúcnosť bez limitov. Výkon bez prestávky, nonstop."

Riešenia EniQ preberajú rutinné úlohy, šetria čas, znižujú náklady a pomáhajú firmám rásť. EniQ sa špecializuje na digitálnych asistentov a automatizáciu procesov, ktoré dokážu zabezpečiť zákaznícku podporu, operatívu aj ďalšiu firemnú komunikáciu 24 hodín denne, 7 dní v týždni.

Za EniQ stojí tím mladých a ambicióznych odborníkov - vývojárov a konzultantov so skúsenosťami s automatizáciou a AI.

=== SLUŽBY ENIQ ===

1. AUTOMATIZÁCIA PROCESOV:
   - Automatizácia opakujúcich sa podnikových úloh
   - Vystavovanie faktúr, párovanie platieb
   - Aktualizácia skladových zásob
   - Generovanie reportov
   - Prenos dát medzi systémami
   - Výrazne zrýchľuje rutinné procesy a obmedzuje manuálnu prácu a chyby

2. CHATBOTI (WEBOVÍ ASISTENTI):
   - Digitálni asistenti na webe pre zvýšenie predajov
   - Zlepšovanie zákazníckej podpory online
   - Rýchle odpovede na otázky návštevníkov
   - Odporúčanie vhodných produktov či riešení
   - Navigácia klienta k dokončeniu nákupu v e-shope
   - Fungujú ako virtuálni predajcovia dostupní 24/7

3. INTERNÍ DIGITÁLNI ASISTENTI:
   - Správa kalendárov
   - Písanie e-mailov
   - Sledovanie úloh
   - Sumarizácia stretnutí
   - Šetria čas zamestnancom
   - Pracujú nonstop bez prestávok

4. RIEŠENIA NA MIERU A KONZULTÁCIE:
   - Vývoj komplexných riešení podľa špecifických potrieb
   - Strategické konzultácie
   - Projekty na mieru podľa cieľov a požiadaviek klienta
   - Analýza potrieb
   - Integrácia do existujúcich systémov

=== KONTAKTNÉ ÚDAJE ===
- Telefón: +420 733 275 349 (vyslov: "plus štyri sta dvadsať, sedemsto tridsaťtri, dva sedem päť, tri štyri deväť")
- E-mail: moucha@eniq.eu (vyslov: "moucha zavináč eniq bodka eu")
- IČO: 23809329 (živnostenské oprávnenie Matěj Moucha)

=== ČASTO KLADENÉ OTÁZKY (FAQ) ===

Q: Čo všetko EniQ vlastne robí?
A: EniQ sa zaoberá vývojom digitálnych asistentov a inteligentných automatizácií pre firmy. Tieto technológie dokážu prevziať širokú škálu firemných činností od zákazníckej podpory a komunikácie až po vnútornú operatívu, a to úplne automaticky 24/7.

Q: Je to vždy na mieru, alebo máte hotové produkty?
A: Riešenia od EniQ sú väčšinou prispôsobené na mieru podľa potrieb zákazníka. Ponúkame strategické konzultácie a vyvíjame projekty presne podľa vašich cieľov a požiadaviek. Niektoré moduly môžu byť pripravené ako základ, ale vždy sú nakonfigurované pre konkrétnu firmu.

Q: Čo je to digitálny asistent a ako mi pomôže?
A: Digitálny asistent je softvér využívajúci AI, ktorý dokáže autonómne vykonávať rôzne úlohy podobne ako ľudský asistent. Prevezme každodenné úlohy - plánovanie kalendára, odpovedanie na e-maily, sledovanie úloh alebo pripravovanie zhrnutí stretnutí. Ušetríte čas a procesy bežia plynule aj mimo pracovnej doby.

Q: Čo presne znamená automatizácia procesov?
A: Ide o nasadenie digitálnych nástrojov, ktoré automaticky vykonávajú opakujúce sa procesy v podniku bez zásahu človeka. Môžeme automatizovať fakturáciu, párovanie platieb, aktualizáciu databáz alebo iné administratívne úkony. Procesy bežia rýchlejšie, sú menej chybové a zamestnanci sa môžu venovať kvalifikovanejšej práci.

Q: Ako funguje chatbot na webe alebo v e-shope?
A: Webový chatbot funguje ako virtuálna podpora či predajca na vašich stránkach. Je neustále k dispozícii a okamžite odpovedá na otázky návštevníkov. Zákazníkovi odporučí vhodný produkt podľa jeho potrieb a prevedie ho procesom nákupu až k dokončeniu objednávky. Zvyšuje pohodlie zákazníkov a pravdepodobnosť úspešného nákupu.

=== PRÍKLADY POZDRAVOV ===
- Ráno (6:00-12:00): "Dobré ráno, tu Alex z EniQ, ako vám môžem pomôcť?"
- Odpoludnie (12:00-18:00): "Dobrý deň, tu Alex z EniQ, ako vám môžem pomôcť?"
- Večer (18:00-22:00): "Dobrý večer, tu Alex z EniQ, ako vám môžem pomôcť?"
- Noc (22:00-6:00): "Dobrý večer, tu Alex z EniQ, ako vám môžem pomôcť?"

Pamätaj: Odpovedaj prirodzene a konverzačne, ako skutočný človek. Si hlas spoločnosti EniQ.
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
