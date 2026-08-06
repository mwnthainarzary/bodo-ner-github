from config import *

from utils import *

from parser import process_sentence


def main():

    df=load_dataset(INPUT_FILE)

    sentences=group_sentences(df)

    for s in sentences:

        process_sentence(s)

        for idx,tag in zip(s["index"],s["tags"]):

            df.at[idx,"bio_tag"]=tag

    save_dataset(df,OUTPUT_FILE)

    print("Completed")


if __name__=="__main__":

    main()