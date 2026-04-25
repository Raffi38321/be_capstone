import re

def bersihkan_teks(teks: str) -> str:
    teks = re.sub(r'[\u2028\u2029\u0085]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks)
    teks = re.sub(r'[^\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF]', '', teks)
    return teks.strip().lower()

def ambil_bagian_skill(text):
    match = re.search(
        r'(skills|keahlian)(.*?)(experience|pengalaman|education|pendidikan|project)',
        text, re.I
    )
    return match.group(2) if match else text