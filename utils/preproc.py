import re

def extract_text(text):
    pattern = r'Krótki opis przedmiotu zamówienia(.*?)4\.2\.6\.\) Główny kod CPV'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ' '