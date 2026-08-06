import re

def clean_token(token: str) -> str:
    """
    Remove punctuation only from beginning/end.
    Preserve:
        120(B)
        No.1383/09
        Officer-In-Charge
        charge-sheet
    """

    token = str(token).strip()

    token = re.sub(r'^[^\w]+', '', token)
    token = re.sub(r'[^\w]+$', '', token)

    return token


def normalize(token: str) -> str:
    return clean_token(token).lower()