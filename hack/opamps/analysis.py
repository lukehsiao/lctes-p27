"""
This script acts as a placeholder to set a threshold for cands >b
and compare those cands with our gold data.
"""
import csv
import logging
import os
import pdb

import numpy as np
from quantiphy import Quantity
from tqdm import tqdm

from hack.opamps.opamp_utils import (
    Score,
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def print_score(score, description="Entity level scores:"):
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


def get_entity_set(file, b=0.0, is_gain=True):
    entities = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            try:
                (doc, val, score) = line
                if float(score) > b:
                    if is_gain:
                        val = f"{float(val)} kHz"
                    else:
                        val = f"{float(val)} uA"
                    entities.add((doc.upper(), Quantity(val)))
            except Exception as e:
                logger.error(f"{e} while getting entity set from {file}.")
    return entities


def get_filenames(entities):
    filenames = set()
    for (doc, val) in entities:
        filenames.add(doc)
    return filenames


def get_filenames_from_file(file):
    filenames = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            filenames.add(line[0])
    return filenames


def get_filenames_from_dir(dirname):
    filenames = set()
    for filename in os.listdir(dirname):
        if not filename.endswith(".pdf") and not filename.endswith(".PDF"):
            logger.warn(f"Invalid filename {filename}, skipping.")
        if filename in filenames:
            logger.warn(f"Duplicate filename {filename}, skipping.")
        filenames.add(filename.replace(".pdf", "").replace(".PDF", ""))
    return filenames


def filter_filenames(entities, filenames):
    result = set()
    for (doc, val) in entities:
        if doc in filenames:
            result.add((doc, val))
    if len(result) == 0:
        logger.debug(
            f"Filtering for {len(get_filenames(entities))} "
            + "entity filenames turned up empty."
        )
    return result


def capitalize_filenames(filenames):
    output = set()
    for filename in filenames:
        output.add(filename.upper())
    return output


def main(
    num=100,
    devfile="gain_dev_probs.csv",
    testfile="gain_test_probs.csv",
    outfile="analysis/gain_analysis_discrepancies.csv",
    is_gain=True,
    debug=False,
):
    # Define file locations
    dirname = os.path.dirname(os.path.abspath(__file__))
    discrepancy_file = os.path.join(dirname, outfile)

    # Dev
    dev_file = os.path.join(dirname, devfile)
    dev_filenames = capitalize_filenames(
        get_filenames_from_file(os.path.join(dirname, "data/dev/filenames.csv"))
    )
    dev_gold_file = os.path.join(dirname, "data/dev/dev_gold.csv")
    dev_gold = filter_filenames(
        get_gold_set(gold=[dev_gold_file], is_gain=is_gain), dev_filenames
    )

    best_dev_score = Score(0, 0, 0, [], [], [])
    best_dev_b = 0
    best_dev_entities = set()

    # Test
    test_file = os.path.join(dirname, testfile)
    test_filenames = capitalize_filenames(
        get_filenames_from_file(os.path.join(dirname, "data/test/filenames.csv"))
    )
    test_gold_file = os.path.join(dirname, "data/test/test_gold.csv")
    test_gold = filter_filenames(
        get_gold_set(gold=[test_gold_file], is_gain=is_gain), test_filenames
    )

    best_test_score = Score(0, 0, 0, [], [], [])
    best_test_b = 0
    best_test_entities = set()

    # Analysis
    filenames_file = os.path.join(dirname, "data/analysis/filenames.csv")
    logger.debug(
        f"Analysis dataset is {len(get_filenames_from_file(filenames_file))}"
        + " filenames long."
    )
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    gold = filter_filenames(
        get_gold_set(gold=[gold_file], is_gain=is_gain),
        capitalize_filenames(get_filenames_from_file(filenames_file)),
    )
    # logger.info(f"Original gold set is {len(get_filenames(gold))} filenames long.")

    best_score = Score(0, 0, 0, [], [], [])
    best_b = 0
    best_entities = set()

    if len(dev_gold) == 0 or len(test_gold) == 0 or len(gold) == 0:
        logger.error("Gold is empty")
        pdb.set_trace()

    logger.info(f"Determining best b...")
    for b in tqdm(np.linspace(0.0, 1, num=num)):
        # Dev
        dev_entities = get_entity_set(dev_file, b=b, is_gain=is_gain)
        dev_score = entity_level_scores(dev_entities, is_gain=is_gain, metric=dev_gold)

        if dev_score.f1 > best_dev_score.f1:
            best_dev_score = dev_score
            best_dev_b = b
            best_dev_entities = dev_entities

        # Test
        test_entities = get_entity_set(test_file, b=b, is_gain=is_gain)
        test_score = entity_level_scores(
            test_entities, is_gain=is_gain, metric=test_gold
        )

        if test_score.f1 > best_test_score.f1:
            best_test_score = test_score
            best_test_b = b
            best_test_entities = test_entities

        # Analysis
        entities = filter_filenames(
            dev_entities.union(test_entities), get_filenames(gold)
        )
        score = entity_level_scores(entities, is_gain=is_gain, metric=gold)

        if score.f1 > best_score.f1:
            best_score = score
            best_b = b
            best_entities = entities

    if debug:
        # test
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
        is_gain=is_gain,
        type="FP",
        outfile=discrepancy_file,
        gold_dic=gold_set_to_dic(gold),
    )
    compare_entities(
        set(best_score.FN),
        is_gain=is_gain,
        type="FN",
        outfile=discrepancy_file,
        append=True,
        entity_dic=gold_set_to_dic(best_entities),
    )
