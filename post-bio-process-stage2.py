import pandas as pd
import re

# -------------------------
# Load BIO dataset
# -------------------------
df = pd.read_csv("bio_dataset_fixed.csv")

df["token"] = df["token"].fillna("").astype(str)
df["bio_tag"] = df["bio_tag"].fillna("O").astype(str)

tokens = df["token"].tolist()
tags = df["bio_tag"].tolist()


def set_entity(start, end, label):
    """Assign BIO tags from start (inclusive) to end (exclusive)."""
    if start >= len(tags):
        return
    tags[start] = f"B-{label}"
    for i in range(start + 1, min(end, len(tags))):
        tags[i] = f"I-{label}"


# -------------------------
# Dictionaries
# -------------------------

DOCUMENTS = {
    "fir", "f.i.r",
    "charge-sheet", "chargesheet", "charge", "sheet",
    "petition", "appeal", "judgment", "judgement",
    "affidavit", "complaint", "charges"
}

LAWS = {
    "ipc", "i.p.c.", "crpc", "cr.p.c.",
    "constitution", "evidence", "act"
}

LEGAL_ROLES = {
    "petitioner",
    "respondent",
    "appellant",
    "accused",
    "complainant",
    "advocate",
    "justice",
    "judge",
    "magistrate",
    "registrar",
    "commissioner"
}

MONTHS = {
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
}

# -------------------------
# Main processing
# -------------------------

i = 0

while i < len(tokens):

    tok = tokens[i]
    low = tok.lower().strip(".,;:()")

    # ------------------------------------------------
    # Preserve existing manual annotations
    # ------------------------------------------------
    if tags[i] != "O":
        i += 1
        continue

    # ------------------------------------------------
    # DOCUMENTS
    # ------------------------------------------------
    if low in DOCUMENTS:
        tags[i] = "B-DOCUMENT"

    # ------------------------------------------------
    # LAW
    # ------------------------------------------------
    elif low in LAWS:
        tags[i] = "B-LAW"

    # ------------------------------------------------
    # SECTION
    # ------------------------------------------------
    elif low in {"section", "sections", "sec", "article"}:

        j = i + 1

        while j < len(tokens):

            nxt = tokens[j]

            if re.match(r"^[0-9A-Za-z().,/()-]+$", nxt):
                j += 1
            else:
                break

        set_entity(i, j, "SECTION")

    # ------------------------------------------------
    # CASE NUMBER
    # ------------------------------------------------
    elif low == "case":

        j = i + 1

        while j < len(tokens):

            nxt = tokens[j]

            if re.search(r"(no|no\.|[0-9/().-])", nxt.lower()):
                j += 1
            else:
                break

        set_entity(i, j, "CASE_NUMBER")

    # ------------------------------------------------
    # FIR NUMBER
    # ------------------------------------------------
    elif low == "fir":

        tags[i] = "B-DOCUMENT"

        if i + 2 < len(tokens):

            if tokens[i+1].lower().startswith("no"):

                tags[i+1] = "I-DOCUMENT"
                tags[i+2] = "I-DOCUMENT"

    # ------------------------------------------------
    # HIGH COURT
    # ------------------------------------------------
    elif low == "high":

        if i + 1 < len(tokens):

            if tokens[i+1].lower() == "court":

                set_entity(i, i+2, "COURT")

    # ------------------------------------------------
    # SUPREME COURT
    # ------------------------------------------------
    elif low == "supreme":

        if i + 1 < len(tokens):

            if tokens[i+1].lower() == "court":

                set_entity(i, i+2, "COURT")

    # ------------------------------------------------
    # POLICE STATION
    # ------------------------------------------------
    elif low == "police":

        if i + 1 < len(tokens):

            if tokens[i+1].lower() == "station":

                set_entity(i, i+2, "POLICE_STATION")

    # ------------------------------------------------
    # P.S.
    # ------------------------------------------------
    elif re.fullmatch(r"p\.?s\.?", low):

        tags[i] = "B-POLICE_STATION"

    # ------------------------------------------------
    # JUSTICE + NAME
    # ------------------------------------------------
    elif low == "justice":

        tags[i] = "B-JUDGE"

        j = i + 1

        while j < len(tokens):

            nxt = tokens[j]

            if re.match(r"^[A-Z][a-zA-Z.-]+$", nxt):
                tags[j] = "I-JUDGE"
                j += 1
            else:
                break

    # ------------------------------------------------
    # DATES
    # ------------------------------------------------
    elif re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", tok):

        tags[i] = "B-DATE"

    elif tok.lower() in MONTHS:

        tags[i] = "B-DATE"

    # ------------------------------------------------
    # DAG NO
    # ------------------------------------------------
    elif low == "dag":

        if i + 2 < len(tokens):

            set_entity(i, i+3, "PROPERTY")

    # ------------------------------------------------
    # PATTA NO
    # ------------------------------------------------
    elif low == "patta":

        if i + 2 < len(tokens):

            set_entity(i, i+3, "PROPERTY")

    # ------------------------------------------------
    # KHATIAN
    # ------------------------------------------------
    elif low == "khatian":

        if i + 2 < len(tokens):

            set_entity(i, i+3, "PROPERTY")

    # ------------------------------------------------
    # Legal Roles
    # ------------------------------------------------
    elif low in LEGAL_ROLES:

        tags[i] = "B-LEGAL_ROLE"

    i += 1


# -------------------------
# Save corrected dataset
# -------------------------

df["bio_tag"] = tags

df.to_csv(
    "bio_dataset_stage2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved -> bio_dataset_stage2.csv")