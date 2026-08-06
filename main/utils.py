import pandas as pd

from config import ENCODING


def load_dataset(path):

    df = pd.read_csv(path,encoding=ENCODING)

    df["token"]=df["token"].fillna("").astype(str)

    df["bio_tag"]=df["bio_tag"].fillna("O").astype(str)

    return df


def save_dataset(df,path):

    df.to_csv(path,index=False,encoding=ENCODING)


def group_sentences(df):

    sentences=[]

    for sid,group in df.groupby("id",sort=False):

        sentences.append({

            "id":sid,

            "tokens":group["token"].tolist(),

            "tags":group["bio_tag"].tolist(),

            "index":group.index.tolist()

        })

    return sentences