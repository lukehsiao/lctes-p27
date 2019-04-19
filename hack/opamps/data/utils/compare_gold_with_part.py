"""
This should read in two gold files, score them against eachother,
and write all discrepancies to an output CSV.
"""
import csv
import logging
import os

from hack.transistors.transistor_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def remove_units(goldfile):
    entities = []
    with open(goldfile, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for line in reader:
            (filename, manuf, part, attr, val) = line

            # Allow the double of a +/- value to be valid also.
            if val.startswith("±"):
                (value, unit) = val.split(" ")
                val = str(2 * float(value[1:]))

            entities.append((filename, manuf, part, attr, val))
    outfile = f"{os.path.dirname(goldfile)}/temp_{goldfile.split('/')[-1]}"
    with open(outfile, "w") as inputcsv:
        writer = csv.writer(inputcsv)
        for line in entities:
            writer.writerow(line)
    return outfile


def print_score(score, entities=None, metric=None):
    logger.info("===================================================")
    logger.info(f"Scoring on {entities} using {metric} as metric")
    logger.info("===================================================")
    logger.info(f"Corpus Precision {score.prec:.3f}")
    logger.info(f"Corpus Recall    {score.rec:.3f}")
    logger.info(f"Corpus F1        {score.f1:.3f}")
    logger.info("---------------------------------------------------")
    logger.info(
        f"TP: {len(score.TP)} " f"| FP: {len(score.FP)} " f"| FN: {len(score.FN)}"
    )
    logger.info("===================================================\n")


if __name__ == "__main__":

    # Compare our gold with Digikey's gold for the analysis set of 66 docs
    # (docs that both we and Digikey have gold labels for)
    # NOTE: We use our gold as gold for this comparison against Digikey
    for attribute in ["typ_supply_current", "typ_gbp"]:
        logger.info(f"Comparing gold labels for {attribute}...")

        dirname = os.path.dirname(__name__)
        outfile = os.path.join(
            dirname, f"../analysis/{attribute}_gold_discrepancies.csv"
        )

        # Us
        our_gold = remove_units(os.path.join(dirname, "../analysis/our_gold.csv"))
        our_gold_set = get_gold_set(gold=[our_gold], attribute=attribute)
        our_gold_dic = gold_set_to_dic(our_gold_set)

        # Digikey
        digikey_gold = remove_units(
            os.path.join(dirname, "../analysis/digikey_gold.csv")
        )
        digikey_gold_set = get_gold_set(gold=[digikey_gold], attribute=attribute)
        digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

        # Score Digikey using our gold as metric
        score = entity_level_scores(
            digikey_gold_set, metric=our_gold_set, attribute=attribute
        )
        logger.info(f"Scores for {attribute}")
        print_score(score, entities=digikey_gold, metric=our_gold)

        # Run final comparison using FN and FP
        compare_entities(
            set(score.FN),
            entity_dic=digikey_gold_dic,
            type="FN",
            gold_dic=our_gold_dic,
            outfile=outfile,
        )
        compare_entities(
            set(score.FP),
            type="FP",
            append=True,
            gold_dic=our_gold_dic,
            outfile=outfile,
        )

        # Remove temp gold files
        os.remove(our_gold)
        os.remove(digikey_gold)
