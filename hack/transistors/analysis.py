"""
This script acts as a placeholder to set a threshold for cands >b
and compare those cands with our gold data.
"""
import csv
import logging
import os
import pickle
from enum import Enum

import numpy as np
from tqdm import tqdm

from hack.transistors.transistor_utils import (
    Score,
    compare_entities,
    entity_level_scores,
    get_gold_set,
    get_implied_parts,
    gold_set_to_dic,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Enum for tracking
class Relation(Enum):
    STG_TEMP_MIN = "stg_temp_min"
    STG_TEMP_MAX = "stg_temp_max"
    POLARITY = "polarity"
    CE_V_MAX = "ce_v_max"


def load_parts_by_doc():
    dirname = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(dirname, "data/parts_by_doc_new.pkl")
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


def capitalize_filenames(filenames):
    output = set()
    for filename in filenames:
        output.add(filename.upper())
    return output


def print_score(score, description):
    logger.info("===================================================")
    logger.info(description)
    logger.info("===================================================")
    logger.info(f"Corpus Precision {score.prec:.3f}")
    logger.info(f"Corpus Recall    {score.rec:.3f}")
    logger.info(f"Corpus F1        {score.f1:.3f}")
    logger.info("---------------------------------------------------")
    logger.info(
        f"TP: {len(score.TP)} " f"| FP: {len(score.FP)} " f"| FN: {len(score.FN)}"
    )
    logger.info("===================================================\n")


def get_entity_set(file, parts_by_doc, b=0.0):
    entities = set()
    errors = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            try:
                (doc, part, val, score) = line
                if float(score) > b:
                    # Add implied parts as well
                    for p in get_implied_parts(part, doc, parts_by_doc):
                        entities.add((doc, p, val))
            except KeyError:
                if doc not in errors:
                    logger.warning(f"{doc} was not found in parts_by_doc.")
                errors.add(doc)
                continue
            except Exception as e:
                logger.error(f"{e} while getting entity set from {file}.")
    return entities


def get_parts(entities):
    parts = set()
    for (doc, part, val) in entities:
        parts.add(part)
    return parts


def get_filenames(entities):
    filenames = set()
    for (doc, part, val) in entities:
        filenames.add(doc)
    return filenames


def print_filenames_to_file(entities, outfile):
    with open(outfile, "w") as outfile:
        writer = csv.writer(outfile)
        for (doc, part, val) in entities:
            writer.writerow([doc])


def get_filenames_from_file(file):
    filenames = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            filenames.add(line[0].upper())
    return filenames


def filter_filenames(entities, filenames):
    result = set()
    for (doc, part, val) in entities:
        if doc in filenames:
            result.add((doc, part, val))
    if len(result) == 0:
        logger.debug(
            f"Filtering for {len(get_filenames(entities))} "
            + "entity filenames turned up empty."
        )
    return result


def main(
    num=100,
    relation=Relation.CE_V_MAX.value,
    devfile="ce_v_max_dev_probs.csv",
    testfile="ce_v_max_test_probs.csv",
    outfile="analysis/ce_v_max_analysis_discrepancies.csv",
    debug=False,
):
    # Define output
    dirname = os.path.dirname(os.path.abspath(__file__))
    discrepancy_file = os.path.join(dirname, outfile)

    # Analysis
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    filenames_file = os.path.join(dirname, "data/analysis/filenames.csv")
    filenames = capitalize_filenames(get_filenames_from_file(filenames_file))
    # logger.info(f"Analysis dataset is {len(filenames)}" + " filenames long.")
    gold = filter_filenames(
        get_gold_set(gold=[gold_file], attribute=relation), filenames
    )
    # logger.info(f"Original gold set is {len(get_filenames(gold))} filenames long.")

    best_score = Score(0, 0, 0, [], [], [])
    best_b = 0
    best_entities = set()

    # Test
    test_file = os.path.join(dirname, testfile)
    test_filenames = capitalize_filenames(
        get_filenames_from_file(os.path.join(dirname, "data/test/filenames.csv"))
    )
    test_goldfile = os.path.join(dirname, "data/test/test_gold.csv")
    test_gold = filter_filenames(
        get_gold_set(gold=[test_goldfile], attribute=relation), test_filenames
    )

    best_test_score = Score(0, 0, 0, [], [], [])
    best_test_b = 0
    best_test_entities = set()

    # Dev
    dev_file = os.path.join(dirname, devfile)
    dev_filenames = capitalize_filenames(
        get_filenames_from_file(os.path.join(dirname, "data/dev/filenames.csv"))
    )
    dev_goldfile = os.path.join(dirname, "data/dev/dev_gold.csv")
    dev_gold = filter_filenames(
        get_gold_set(gold=[dev_goldfile], attribute=relation), dev_filenames
    )

    best_dev_score = Score(0, 0, 0, [], [], [])
    best_dev_b = 0
    best_dev_entities = set()

    # Iterate over `b` values
    logger.info(f"Determining best b...")
    parts_by_doc = load_parts_by_doc()
    for b in tqdm(np.linspace(0, 1, num=num)):
        # Dev and Test
        dev_entities = get_entity_set(dev_file, parts_by_doc, b=b)
        test_entities = get_entity_set(test_file, parts_by_doc, b=b)

        # Analysis (combo of dev and test)
        entities = filter_filenames(
            dev_entities.union(test_entities), get_filenames_from_file(filenames_file)
        )

        # Score entities against gold data and generate comparison CSV
        dev_score = entity_level_scores(
            dev_entities, attribute=relation, docs=dev_filenames
        )
        test_score = entity_level_scores(
            test_entities, attribute=relation, docs=test_filenames
        )
        score = entity_level_scores(entities, attribute=relation, docs=filenames)

        if dev_score.f1 > best_dev_score.f1:
            best_dev_score = dev_score
            best_dev_b = b
            best_dev_entities = dev_entities

        if test_score.f1 > best_test_score.f1:
            best_test_score = test_score
            best_test_b = b
            best_test_entities = test_entities

        if score.f1 > best_score.f1:
            best_score = score
            best_b = b
            best_entities = entities

    if debug:
        # Test
        logger.info("Scoring for test set...")
        logger.info(
            f"Entity set is {len(get_filenames(best_test_entities))} filenames long."
        )
        logger.info(f"Gold set is {len(get_filenames(test_gold))} filenames long.")
        print_score(
            best_test_score,
            description=f"Scoring on cands > {best_test_b:.3f} "
            + "against our gold labels.",
        )

        # Dev
        logger.info("Scoring for dev set...")
        logger.info(
            f"Entity set is {len(get_filenames(best_dev_entities))} filenames long."
        )
        logger.info(f"Gold set is {len(get_filenames(dev_gold))} filenames long.")
        print_score(
            best_dev_score,
            description=f"Scoring on cands > {best_dev_b:.3f} against our gold labels.",
        )

        logger.info("Scoring for analysis set...")
    # Analysis
    # logger.info(f"Entity set is {len(get_filenames(best_entities))} filenames long.")
    # logger.info(f"Gold set is {len(get_filenames(gold))} filenames long.")
    print_score(
        best_score,
        description=f"Scoring on cands > {best_b:.3f} against our gold labels.",
    )

    compare_entities(
        set(best_score.FP),
        attribute=relation,
        type="FP",
        outfile=discrepancy_file,
        gold_dic=gold_set_to_dic(gold),
    )
    compare_entities(
        set(best_score.FN),
        attribute=relation,
        type="FN",
        outfile=discrepancy_file,
        append=True,
        entity_dic=gold_set_to_dic(best_entities),
    )
