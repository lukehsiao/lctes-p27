"""
This should read in two gold files, score them against eachother,
and write all discrepancies to an output CSV.
"""
import logging
import os

from hack.opamps.opamp_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    for attribute in ["current", "gain"]:
        if attribute == "gain":
            is_gain = True
        else:
            is_gain = False

        logger.info(f"Comparing gold labels for {attribute}...")

        dirname = os.path.dirname(__name__)
        outfile = os.path.join(
            dirname, f"../../analysis/{attribute}_digikey_discrepancies.csv"
        )

        # Us
        our_gold = os.path.join(dirname, "../analysis/our_gold.csv")
        our_gold_set = get_gold_set(gold=[our_gold], is_gain=is_gain)
        our_gold_dic = gold_set_to_dic(our_gold_set)

        # Digikey
        digikey_gold = os.path.join(dirname, "../analysis/digikey_gold.csv")
        digikey_gold_set = get_gold_set(gold=[digikey_gold], is_gain=is_gain)
        digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

        # Score Digikey using our gold as metric
        score = entity_level_scores(
            digikey_gold_set, metric=our_gold_set, is_gain=is_gain
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
        # NOTE: We only care about the entity_dic for FN as they are the ones
        # where we want to know what Digikey does have for manual evalutation.
        # We already know what we have (as that was the FN that Digikey missed)
        # so all we care about is what Digikey actually does have for a doc.
