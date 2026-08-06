import pandas as pd
import re

# ----------------------------
# Load BIO dataset
# ----------------------------
df = pd.read_csv("bio_ner-eng-dataset-v2.csv")

# Replace NaN
df["token"] = df["token"].fillna("")
df["bio_tag"] = df["bio_tag"].fillna("O")

tokens = df["token"].tolist()
tags = df["bio_tag"].tolist()

# ----------------------------
# Utility
# ----------------------------

def set_entity(start, length, label):
    tags[start] = f"B-{label}"
    for j in range(1, length):
        if start + j < len(tags):
            tags[start+j] = f"I-{label}"


# ----------------------------
# Dictionaries
# ----------------------------

DOCUMENTS = {
    "fir",
    "charge-sheet",
    "chargesheet",
    "charge",
    "judgment",
    "judgement",
    "appeal",
    "petition",
    "affidavit"
}

LAWS = {
    "ipc",
    "i.p.c.",
    "crpc",
    "cr.p.c.",
    "constitution"
}

COURTS = [
    ["supreme","court"],
    ["high","court"]
]

POLICE = [
    ["police","station"],
    ["p.s."]
]

DESIGNATIONS = {
    "judge",
    "justice",
    "advocate",
    "petitioner",
    "respondent",
    "appellant",
    "accused",
    "complainant",
    "officer-in-charge",
    "o/c"
}

# ----------------------------
# Main loop
# ----------------------------

i = 0

while i < len(tokens):

    token = str(tokens[i]).strip()
    low = token.lower()

    # -------------------------
    # DOCUMENT
    # -------------------------

    if low in DOCUMENTS:
        tags[i] = "B-DOCUMENT"

    # -------------------------
    # LAW
    # -------------------------

    elif low in LAWS:
        tags[i] = "B-LAW"

    # -------------------------
    # SECTION
    # -------------------------

    elif low in ["section","sections","sec.","article"]:
        tags[i] = "B-SECTION"

        j = i + 1

        while j < len(tokens):

            nxt = str(tokens[j])

            if re.match(r"^[0-9A-Za-z().,/()-]+$", nxt):
                tags[j] = "I-SECTION"
                j += 1
            else:
                break

    # -------------------------
    # CASE NUMBER
    # -------------------------

    elif low == "case":

        tags[i] = "B-CASE_NUMBER"

        j = i + 1

        while j < len(tokens):

            nxt = str(tokens[j])

            if re.search(r"no|[0-9/()-]", nxt.lower()):
                tags[j] = "I-CASE_NUMBER"
                j += 1
            else:
                break

    # -------------------------
    # COURTS
    # -------------------------

    for pattern in COURTS:

        if i + len(pattern) <= len(tokens):

            seq = [x.lower() for x in tokens[i:i+len(pattern)]]

            if seq == pattern:

                set_entity(i, len(pattern), "COURT")

    # -------------------------
    # POLICE STATION
    # -------------------------

    if low == "police":

        if i + 1 < len(tokens):

            if tokens[i+1].lower() == "station":

                tags[i] = "B-POLICE_STATION"
                tags[i+1] = "I-POLICE_STATION"

    # -------------------------
    # OFFICER IN CHARGE
    # -------------------------

    if low in DESIGNATIONS:

        tags[i] = "B-DESIGNATION"

    # -------------------------
    # DATES
    # -------------------------

    if re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", token):

        tags[i] = "B-DATE"

    # -------------------------
    # DAG NO
    # -------------------------

    if low == "dag":

        tags[i] = "B-PROPERTY"

        if i + 2 < len(tokens):

            tags[i+1] = "I-PROPERTY"
            tags[i+2] = "I-PROPERTY"

    # -------------------------
    # PATTA NO
    # -------------------------

    if low == "patta":

        tags[i] = "B-PROPERTY"

        if i + 2 < len(tokens):

            tags[i+1] = "I-PROPERTY"
            tags[i+2] = "I-PROPERTY"

    i += 1


# ----------------------------
# Save
# ----------------------------

df["bio_tag"] = tags

df.to_csv("bio_dataset_fixed.csv",
          index=False,
          encoding="utf-8-sig")

print("Saved -> bio_dataset_fixed.csv")