import pandas as pd
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

labels = [
    "Person",
    "Court",
    "Police Station",
    "Case Number",
    "Law",
    "Section",
    "Document",
    "Date"
]

df = pd.read_csv("ner-data-only-eng.csv", encoding="cp1252")

results = []

for _, row in df.iterrows():
    text = str(row["judgment"])
    entities = model.predict_entities(text, labels)

    for entity in entities:
        results.append({
            "id": row["id"],
            # "judgment": text,
            "entity_text": entity["text"],
            "entity_label": entity["label"],
            "score": round(entity["score"], 4)
        })


# Save to CSV
result_df = pd.DataFrame(results)
result_df.to_csv("ner_results_without_judgment.csv", index=False, encoding="utf-8-sig")


print("NER results saved successfully!")
