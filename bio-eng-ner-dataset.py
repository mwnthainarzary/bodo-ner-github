import pandas as pd
from gliner import GLiNER

# Load GLiNER model
model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

labels = [
    "PERSON",
    "COURT",
    "JUDGE",
    "LAW",
    "SECTION",
    "CASE_NUMBER",
    "DOCUMENT",
    "POLICE_STATION",
    "DATE",
    "LOCATION",
    "ORGANIZATION",
    "PROPERTY",
    "DESIGNATION",
]

# Read dataset
df = pd.read_csv("ner-data-only-eng.csv", encoding="cp1252")


bio_results = []

for _, row in df.iterrows():

    doc_id = row["id"]
    text = str(row["judgment"])

    # Extract entities
    entities = model.predict_entities(text, labels)

    # Initialize all tokens as O
    tokens = text.split()
    bio_tags = ["O"] * len(tokens)

    # Assign BIO tags
    for ent in entities:

        ent_tokens = ent["text"].split()
        n = len(ent_tokens)

        for i in range(len(tokens) - n + 1):

            if tokens[i:i+n] == ent_tokens:

                bio_tags[i] = f"B-{ent['label']}"

                for j in range(1, n):
                    bio_tags[i+j] = f"I-{ent['label']}"

                break

    # Save token-level annotations
    for token, tag in zip(tokens, bio_tags):
        bio_results.append({
            "id": doc_id,
            "token": token,
            "bio_tag": tag
        })

    # Blank line between sentences/documents
    bio_results.append({
        "id": "",
        "token": "",
        "bio_tag": ""
    })

# Save BIO dataset
bio_df = pd.DataFrame(bio_results)
bio_df.to_csv("bio_ner-eng-dataset-v2.csv", index=False, encoding="utf-8-sig")

print("BIO dataset saved to bio_dataset.csv")