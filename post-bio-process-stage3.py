import re
import pandas as pd

# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "bio_dataset_fixed.csv"
OUTPUT_FILE = "bio_dataset_stage4.csv"

# ==========================================================
# Gazetteers
# ==========================================================

DOCUMENTS = {
    "fir", "chargesheet", "charge-sheet",
    "judgment", "judgement", "order",
    "decree", "petition", "appeal",
    "complaint", "affidavit",
    "application", "revision",
    "plaint", "notice", "memo"
}

LAWS = {
    "ipc",
    "i.p.c",
    "crpc",
    "cr.p.c",
    "constitution",
    "constitution of india",
    "indian penal code",
    "code of criminal procedure",
    "indian evidence act",
    "motor vehicles act",
    "companies act",
    "arms act",
    "ndps act",
    "information technology act"
}

LEGAL_ROLES = {
    "petitioner",
    "respondent",
    "appellant",
    "plaintiff",
    "defendant",
    "accused",
    "complainant",
    "informant",
    "advocate",
    "justice",
    "judge",
    "chief justice",
    "magistrate",
    "registrar",
    "commissioner",
    "public prosecutor",
    "government advocate",
    "senior advocate",
    "officer-in-charge",
    "officer in charge",
    "o/c"
}

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
}

PROPERTY_WORDS = {
    "dag",
    "patta",
    "khatian",
    "survey"
}

COURTS = [
    ["supreme","court"],
    ["high","court"],
    ["gauhati","high","court"],
    ["delhi","high","court"],
    ["bombay","high","court"],
    ["calcutta","high","court"],
    ["madras","high","court"]
]

# ==========================================================
# Regex
# ==========================================================

SECTION_PATTERN = re.compile(
    r'^\d+(\([A-Za-z0-9]+\))?([/-]\d+(\([A-Za-z0-9]+\))?)*$'
)

CASE_PATTERN = re.compile(
    r'^(no|no\.|[0-9]+(/[0-9]+)+|[0-9]+)$'
)

DATE_PATTERN = re.compile(
    r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'
)

# ==========================================================
# Helper functions
# ==========================================================

def normalize(token):
    token = str(token).strip()

    # Remove punctuation only at beginning/end
    token = re.sub(r'^[^\w]+', '', token)
    token = re.sub(r'[^\w]+$', '', token)

    return token.lower()


def mark(tags, start, end, label):

    tags[start] = f"B-{label}"

    for i in range(start + 1, end):
        tags[i] = f"I-{label}"


# ==========================================================
# Load
# ==========================================================

df = pd.read_csv(INPUT_FILE)

df["token"] = df["token"].fillna("").astype(str)
df["bio_tag"] = df["bio_tag"].fillna("O").astype(str)

tokens = df["token"].tolist()
tags = df["bio_tag"].tolist()

normalized = [normalize(x) for x in tokens]

# ==========================================================
# Main loop
# ==========================================================

i = 0

while i < len(tokens):


    if tags[i] != "O":
        continue

    tok = normalized[i]

    # --------------------------
    # DOCUMENT
    # --------------------------

    if tok in DOCUMENTS:

        tags[i] = "B-DOCUMENT"
        continue

    # --------------------------
    # LAW
    # --------------------------

    if tok in LAWS:

        tags[i] = "B-LAW"
        continue

    # --------------------------
    # LEGAL ROLE
    # --------------------------

    if tok in LEGAL_ROLES:

        tags[i] = "B-LEGAL_ROLE"
        continue

    # --------------------------
    # DATE
    # --------------------------

    if DATE_PATTERN.fullmatch(tok) or tok in MONTHS:

        tags[i] = "B-DATE"
        continue

    # --------------------------
    # PROPERTY
    # --------------------------

    if tok in PROPERTY_WORDS:

        mark(tags, i, min(i+3, len(tags)), "PROPERTY")
        continue

    # --------------------------
    # SECTION
    # --------------------------

   
    if tok in {"section", "sections", "article"}:

        tags[i] = "B-SECTION"

        # Only ONE token after Section can be part of the section number
        if i + 1 < len(tokens):

            nxt = normalize(tokens[i + 1])

            if SECTION_PATTERN.fullmatch(nxt):
                tags[i + 1] = "I-SECTION"

        continue

     # --------------------------
    # IPC
    # --------------------------
    
    if tok in {"ipc","i.p.c","crpc","cr.p.c"}:

        tags[i] = "B-LAW"
        continue

    # --------------------------
    # CASE NUMBER
    # --------------------------

    if tok == "case":

        tags[i] = "B-CASE_NUMBER"

        if i + 1 < len(tokens):

            if normalize(tokens[i + 1]) in {"no","no."}:

                tags[i + 1] = "I-CASE_NUMBER"

                if i + 2 < len(tokens):

                    tags[i + 2] = "I-CASE_NUMBER"

        continue

    # --------------------------
    # COURTS
    # --------------------------

    for court in COURTS:

        n = len(court)

        if normalized[i:i+n] == court:

            mark(tags, i, i+n, "COURT")
    i += 2
    continue
# ==========================================================
# Police Station
# ==========================================================

for i in range(len(tokens)-1):

    if normalized[i] == "police" and normalized[i+1] == "station":

        mark(tags, i, i+2, "POLICE_STATION")

# ==========================================================
# Justice Name
# ==========================================================

for i in range(len(tokens)-1):

    if normalized[i] == "justice":

        tags[i] = "B-JUDGE"

        j = i + 1

        while j < len(tokens):

            if re.fullmatch(r"[A-Z][A-Za-z.-]+", tokens[j]):

                tags[j] = "I-JUDGE"
                j += 1

            else:
                break

# ==========================================================
# BIO Validation
# ==========================================================

previous = "O"

for i in range(len(tags)):

    tag = tags[i]

    if tag.startswith("I-"):

        entity = tag[2:]

        if previous not in {

            f"B-{entity}",

            f"I-{entity}"

        }:

            tags[i] = f"B-{entity}"

    previous = tags[i]

# ==========================================================
# Save
# ==========================================================

df["bio_tag"] = tags

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Finished.")
print("Saved:", OUTPUT_FILE)