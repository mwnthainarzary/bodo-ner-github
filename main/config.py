from pathlib import Path

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_FILE = BASE_DIR / "input" / "bio_dataset.csv"

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_FILE = OUTPUT_DIR / "bio_dataset_stage1.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------
# Labels
# -------------------------------------------------------

LABELS = {

    "PERSON",

    "JUDGE",

    "COURT",

    "POLICE_STATION",

    "DOCUMENT",

    "CASE_NUMBER",

    "LAW",

    "SECTION",

    "DATE",

    "LOCATION",

    "ORGANIZATION",

    "PROPERTY",

    "LEGAL_ROLE"

}

# -------------------------------------------------------
# Misc
# -------------------------------------------------------

ENCODING = "utf-8-sig"