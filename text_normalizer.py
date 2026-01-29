"""
Text Normalizer for TTS.
Converts numbers, abbreviations, and special characters to spoken form in Czech.
"""
import re
from typing import Dict, List, Tuple


# Czech number words
ONES = {
    0: "nula", 1: "jedna", 2: "dva", 3: "tři", 4: "čtyři",
    5: "pět", 6: "šest", 7: "sedm", 8: "osm", 9: "devět"
}

TEENS = {
    10: "deset", 11: "jedenáct", 12: "dvanáct", 13: "třináct",
    14: "čtrnáct", 15: "patnáct", 16: "šestnáct", 17: "sedmnáct",
    18: "osmnáct", 19: "devatenáct"
}

TENS = {
    2: "dvacet", 3: "třicet", 4: "čtyřicet", 5: "padesát",
    6: "šedesát", 7: "sedmdesát", 8: "osmdesát", 9: "devadesát"
}

HUNDREDS = {
    1: "sto", 2: "dvě stě", 3: "tři sta", 4: "čtyři sta",
    5: "pět set", 6: "šest set", 7: "sedm set", 8: "osm set", 9: "devět set"
}

THOUSANDS = {
    1: "tisíc", 2: "dva tisíce", 3: "tři tisíce", 4: "čtyři tisíce"
}

# Common abbreviations and their spoken forms
# NOTE: Single letter units (m, l, g) are NOT included here to avoid replacing
# them in the middle of words. They should only be replaced in specific contexts
# like "5 m" which is handled by measurement patterns.
ABBREVIATIONS: Dict[str, str] = {
    "např.": "například",
    "apod.": "a podobně",
    "atd.": "a tak dále",
    "tj.": "to jest",
    "tzn.": "to znamená",
    "resp.": "respektive",
    "cca": "cirka",
    "cca.": "cirka",
    "viz": "viz",
    "vs.": "versus",
    "min.": "minut",
    "max.": "maximálně",
    "hod.": "hodin",
    "s.r.o.": "s r o",
    "a.s.": "a s",
    "Kč": "korun",
    "CZK": "korun",
    "EUR": "eur",
    "USD": "dolarů",
    "tel.": "telefon",
    "č.": "číslo",
    "str.": "strana",
    "obr.": "obrázek",
    "tab.": "tabulka",
    "pozn.": "poznámka",
    "př.": "příklad",
    "vč.": "včetně",
    "MHD": "M H D",
    "DIČ": "D I Č",
    "IČO": "I Č O",
    "www": "w w w",
    "@": "zavináč",
}

# Measurement unit patterns (only when following numbers)
MEASUREMENT_PATTERNS: List[Tuple[str, str]] = [
    (r"(\d+)\s*km\b", r"\1 kilometrů"),
    (r"(\d+)\s*m\b", r"\1 metrů"),
    (r"(\d+)\s*cm\b", r"\1 centimetrů"),
    (r"(\d+)\s*mm\b", r"\1 milimetrů"),
    (r"(\d+)\s*kg\b", r"\1 kilogramů"),
    (r"(\d+)\s*g\b", r"\1 gramů"),
    (r"(\d+)\s*l\b", r"\1 litrů"),
    (r"(\d+)\s*ml\b", r"\1 mililitrů"),
    (r"(\d+)\s*%", r"\1 procent"),
]

# Currency patterns
CURRENCY_PATTERNS: List[Tuple[str, str]] = [
    (r"(\d+)\s*Kč", r"\1 korun"),
    (r"(\d+)\s*CZK", r"\1 korun"),
    (r"(\d+)\s*EUR", r"\1 eur"),
    (r"(\d+)\s*€", r"\1 eur"),
    (r"(\d+)\s*USD", r"\1 dolarů"),
    (r"\$(\d+)", r"\1 dolarů"),
]


def number_to_words(n: int) -> str:
    """Convert a number to Czech words."""
    if n < 0:
        return "mínus " + number_to_words(-n)
    
    if n == 0:
        return ONES[0]
    
    if n < 10:
        return ONES[n]
    
    if n < 20:
        return TEENS[n]
    
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return TENS[tens]
        return f"{TENS[tens]} {ONES[ones]}"
    
    if n < 1000:
        hundreds, remainder = divmod(n, 100)
        if remainder == 0:
            return HUNDREDS[hundreds]
        return f"{HUNDREDS[hundreds]} {number_to_words(remainder)}"
    
    if n < 10000:
        thousands, remainder = divmod(n, 1000)
        if thousands <= 4:
            thousands_word = THOUSANDS[thousands]
        elif thousands < 10:
            thousands_word = f"{ONES[thousands]} tisíc"
        else:
            thousands_word = f"{number_to_words(thousands)} tisíc"
        
        if remainder == 0:
            return thousands_word
        return f"{thousands_word} {number_to_words(remainder)}"
    
    if n < 1000000:
        thousands, remainder = divmod(n, 1000)
        thousands_word = f"{number_to_words(thousands)} tisíc"
        if remainder == 0:
            return thousands_word
        return f"{thousands_word} {number_to_words(remainder)}"
    
    # For very large numbers, just read digits
    return " ".join(ONES[int(d)] for d in str(n))


def normalize_number(match: re.Match) -> str:
    """Convert matched number to words."""
    number_str = match.group(0)
    
    # Handle decimal numbers
    if "," in number_str or "." in number_str:
        number_str = number_str.replace(",", ".")
        parts = number_str.split(".")
        if len(parts) == 2:
            whole = int(parts[0]) if parts[0] else 0
            decimal = parts[1]
            whole_words = number_to_words(whole)
            decimal_words = " ".join(ONES[int(d)] for d in decimal)
            return f"{whole_words} celých {decimal_words}"
    
    # Handle regular integers
    try:
        number = int(number_str.replace(" ", "").replace(",", ""))
        return number_to_words(number)
    except ValueError:
        return number_str


def normalize_time(match: re.Match) -> str:
    """Convert time format to spoken form."""
    hours = int(match.group(1))
    minutes = int(match.group(2))
    
    hours_word = number_to_words(hours)
    
    if minutes == 0:
        return f"{hours_word} hodin"
    
    minutes_word = number_to_words(minutes)
    return f"{hours_word} hodin {minutes_word} minut"


def normalize_date(match: re.Match) -> str:
    """Convert date format to spoken form."""
    day = int(match.group(1))
    month = int(match.group(2))
    
    months = {
        1: "ledna", 2: "února", 3: "března", 4: "dubna",
        5: "května", 6: "června", 7: "července", 8: "srpna",
        9: "září", 10: "října", 11: "listopadu", 12: "prosince"
    }
    
    day_word = number_to_words(day)
    month_word = months.get(month, str(month))
    
    if match.lastindex == 3:
        year = int(match.group(3))
        year_word = number_to_words(year)
        return f"{day_word} {month_word} {year_word}"
    
    return f"{day_word} {month_word}"


def normalize_phone(match: re.Match) -> str:
    """Convert phone number to spoken form with pauses."""
    phone = match.group(0)
    # Remove common separators
    digits = re.sub(r"[\s\-\(\)\+]", "", phone)
    # Group into triplets and read
    result = []
    for i in range(0, len(digits), 3):
        group = digits[i:i+3]
        group_words = " ".join(ONES[int(d)] for d in group)
        result.append(group_words)
    return ", ".join(result)


def normalize_text(text: str) -> str:
    """
    Normalize text for natural TTS output.
    Converts numbers, abbreviations, and special formats to spoken Czech.
    """
    if not text:
        return text
    
    result = text
    
    # Replace abbreviations (case-insensitive for some)
    for abbr, full in ABBREVIATIONS.items():
        # Case-sensitive replacement
        result = result.replace(abbr, full)
        # Also try uppercase version
        result = result.replace(abbr.upper(), full)
    
    # Replace measurement patterns (e.g., "5 m" -> "5 metrů")
    for pattern, replacement in MEASUREMENT_PATTERNS:
        result = re.sub(pattern, replacement, result)
    
    # Replace currency patterns first (before general numbers)
    for pattern, replacement in CURRENCY_PATTERNS:
        result = re.sub(pattern, replacement, result)
    
    # Replace time patterns (HH:MM)
    result = re.sub(r"(\d{1,2}):(\d{2})", normalize_time, result)
    
    # Replace date patterns (DD.MM. or DD.MM.YYYY)
    result = re.sub(r"(\d{1,2})\.(\d{1,2})\.(\d{4})?", normalize_date, result)
    
    # Replace phone numbers (various formats)
    result = re.sub(r"\+?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}", normalize_phone, result)
    
    # Replace standalone numbers (not part of other patterns)
    result = re.sub(r"\b\d+([,\.]\d+)?\b", normalize_number, result)
    
    # Clean up multiple spaces
    result = re.sub(r"\s+", " ", result)
    
    # Remove any remaining special characters that might cause issues
    result = result.replace("*", "")
    result = result.replace("#", "")
    result = result.replace("_", " ")
    
    return result.strip()


# Quick test
if __name__ == "__main__":
    test_cases = [
        "Cena je 150 EUR",
        "Volejte na 123 456 789",
        "Schůzka je v 14:30",
        "Datum: 25.12.2024",
        "Sleva 20%",
        "Máme 3 produkty za 1500 Kč",
        "Např. tato služba stojí cca 200 EUR měsíčně",
    ]
    
    for text in test_cases:
        print(f"Input:  {text}")
        print(f"Output: {normalize_text(text)}")
        print()
