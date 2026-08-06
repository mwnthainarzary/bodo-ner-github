COURTS={

"gauhati high court",

"supreme court"

}


def parse_gazetteer(tokens,tags):

    words=[x.lower() for x in tokens]

    sentence=" ".join(words)

    if "gauhati high court" in sentence:

        for i in range(len(words)-2):

            if words[i:i+3]==["gauhati","high","court"]:

                tags[i]="B-COURT"

                tags[i+1]="I-COURT"

                tags[i+2]="I-COURT"

    return tags