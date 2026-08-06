import re

SECTION_REGEX=re.compile(

r'^\d+(\([A-Za-z0-9]+\))?([/-]\d+(\([A-Za-z0-9]+\))?)*$'

)

DATE_REGEX=re.compile(

r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'

)


def parse_regex(tokens,tags):

    i=0

    while i<len(tokens):

        tok=tokens[i].lower()

        # ----------------------
        # SECTION
        # ----------------------

        if tok in {"section","sections","article"}:

            tags[i]="B-SECTION"

            if i+1<len(tokens):

                if SECTION_REGEX.fullmatch(tokens[i+1]):

                    tags[i+1]="I-SECTION"

            i+=2

            continue

        # ----------------------
        # DATE
        # ----------------------

        if DATE_REGEX.fullmatch(tokens[i]):

            tags[i]="B-DATE"

        i+=1

    return tags