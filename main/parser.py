from regex_parser import parse_regex

from gazetteer import parse_gazetteer


def process_sentence(sentence):

    tokens=sentence["tokens"]

    tags=sentence["tags"]

    tags=parse_regex(tokens,tags)

    tags=parse_gazetteer(tokens,tags)

    sentence["tags"]=tags

    return sentence