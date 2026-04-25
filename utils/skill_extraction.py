import re

SKILL_LIST = [
    "python","java","c++","javascript","html","css",
    "react","node js","tensorflow","keras","pandas","numpy",
    "sql","mysql","postgresql",
    "machine learning","deep learning","nlp"
]

SYNONYM = {
    "js": "javascript",
    "py": "python",
    "ml": "machine learning"
}

def normalize_synonym(text):
    words = text.split()
    return " ".join([SYNONYM.get(w, w) for w in words])

def extract_skills(text):
    found = []
    for skill in SKILL_LIST:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found.append(skill)
    return list(set(found))