import os
import spacy
import random
import json
from spacy.matcher import Matcher
from spacy.util import filter_spans
from multiprocessing import Pool
import argparse
import pandas as pd
from tqdm import tqdm
from copy import copy


def check_same(trans: str, target: str):
    """
    check if the transformed string is equal to the target string.
    """
    trans = trans.replace(".", "").replace(",", "")
    target = target.replace(".", "").replace(",", "")

    if " ".join(trans.strip().split()) == " ".join(target.strip().split()):
        return True
    else:
        return False


def swapper(text, idxs):
    """
    given the indexes, swap position in the list
    """
    mini, maxi = random.sample(idxs, 2)
    temp = text[mini]
    text[mini] = text[maxi]
    text[maxi] = temp
    return " ".join(text)


def inner_pos_and_tag_finder(text, result, token, ex):
    noun_idx = [i for i, (tag) in enumerate(result) if tag in [token]]
    if len(noun_idx) > 1:
        text = swapper(text, noun_idx)

        if not check_same(text, ex):
            return text
        else:
            return None
    else:
        return None


def global_find_hard_negatives(sentence):
    """
    Function that computes hard negatives
    """
    doc = nlp(sentence)
    text = []
    tag_result = []
    pos_result = []

    for token in doc:
        text.append(token.text)
        tag_result.append(token.tag_)
        pos_result.append(token.pos_)

    # we put all hard negatives in a list
    novel_hard_negatives = []
    novel_hard_negatives.append(shuffle_noun_phrase(doc, sentence))
    novel_hard_negatives.append(shuffle_verb_phrases(doc, sentence))
    novel_hard_negatives.append(shuffler(copy(text), tag_result, sentence, "NN"))
    novel_hard_negatives.append(shuffler(copy(text), tag_result, sentence, "NNS"))
    novel_hard_negatives.append(shuffler(copy(text), pos_result, sentence, "ADV"))
    novel_hard_negatives.append(shuffler(copy(text), pos_result, sentence, "ADJ"))

    return novel_hard_negatives


def shuffle_verb_phrases(parsed, sent):
    pattern = [
        {"POS": "VERB", "OP": "?"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "AUX", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
    ]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])

    # call the matcher to find matches
    matches = matcher(parsed)
    spans = [parsed[start:end] for _, start, end in matches]

    nnc = filter_spans(spans)

    if len(nnc) > 1:
        sampled = random.sample(nnc, 2)
        new_sent = sent.replace(sampled[0].text, "123X456")
        new_sent = new_sent.replace(sampled[1].text, sampled[0].text)
        new_sent = new_sent.replace("123X456", sampled[1].text)

        if not check_same(new_sent, sent):
            return new_sent.lower()
        else:
            return None

    else:
        return None


def shuffle_noun_phrase(parsed, sent):
    nnc = list(parsed.noun_chunks)
    nnc = list(filter(lambda x: len(x.text.split()) > 2, nnc))

    if len(nnc) > 1:

        sampled = random.sample(nnc, 2)
        new_sent = sent.replace(sampled[0].text, "123X456")
        new_sent = new_sent.replace(sampled[1].text, sampled[0].text)
        new_sent = new_sent.replace("123X456", sampled[1].text)

        if not check_same(new_sent, sent):
            return new_sent.lower()
        else:
            return None

    else:
        return None


def shuffler(text, result, ex, pos_or_tag):
    take = inner_pos_and_tag_finder(text, result, pos_or_tag, ex)
    return take


def extract_dataset(coco_path, dataset_csv_path):
    """
    Extracts the COCO dataset to a csv format
    :param coco_path: path to the training split of karpathy coco training split
    :param dataset_csv_path: path to save the csv file
    :return:
    """
    data = json.load(open(coco_path))
    df = pd.DataFrame(data)
    # Keep a single caption for each image.
    df = df.drop_duplicates(subset=["image_id"], keep="first")
    df.to_csv(dataset_csv_path, index=False, sep="\t")


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco-train-path",
        default="/oak/stanford/groups/jamesz/fede/coco_karpathy/coco_karpathy_train.json",
        type=str,
        help="Path to the json file for the karpathy splits for COCO.",
    )
    parser.add_argument(
        "--dataset-csv",
        default="/oak/stanford/groups/jamesz/merty/coco.csv",
        type=str,
        help="Path to a csv file with captions and image ids/paths.",
    )
    parser.add_argument(
        "--save_path",
        default="/oak/stanford/groups/jamesz/merty/coco_negclip.csv",
        type=str,
        help="Path to a csv file with the additional negative captions and image ids/paths.",
    )
    parser.add_argument("--cores", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = config()

    # If the csv file does not exist, use the coco file to generate it.
    if not os.path.exists(args.dataset_csv):
        extract_dataset(
            coco_path=args.coco_train_path, dataset_csv_path=args.dataset_csv
        )

    # Load the spacy module for the POS tagging
    nlp = spacy.load("en_core_web_lg")

    # Generate negative captions.
    train_coco = pd.read_csv(args.dataset_csv, sep="\t")
    print(train_coco.head())
    with Pool(args.cores) as p:
        r = list(
            tqdm(
                p.imap(global_find_hard_negatives, train_coco["caption"].tolist()),
                total=len(train_coco),
            )
        )

    train_coco["neg_caption"] = r
    train_coco["neg_caption"] = train_coco["neg_caption"].apply(
        lambda lst: [x for x in lst if x is not None]
    )
    train_coco["neg_caption"] = train_coco["neg_caption"].apply(
        lambda x: None if x == [] else x
    )
    train_coco = train_coco.dropna()
    train_coco.to_csv(args.save_path, index=False, sep="\t")
