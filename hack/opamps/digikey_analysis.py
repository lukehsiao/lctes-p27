"""
This should read in two gold files, score them against eachother,
and write all discrepancies to an output CSV.
"""
import logging
import os

from hack.opamps.analysis import print_score
from hack.opamps.opamp_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(is_gain=False):
    if is_gain:
        attribute = "gain"
    else:
        attribute = "current"

    dirname = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(dirname, f"analysis/{attribute}_digikey_discrepancies.csv")

    # Us
    our_gold = os.path.join(dirname, "data/analysis/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], is_gain=is_gain)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Digikey
    digikey_gold = os.path.join(dirname, "data/analysis/digikey_gold.csv")
    digikey_gold_set = get_gold_set(gold=[digikey_gold], is_gain=is_gain)
    digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

    # Score Digikey using our gold as metric
    score = entity_level_scores(digikey_gold_set, metric=our_gold_set, is_gain=is_gain)
    # logger.info(f"Scores for {attribute}...")
    print_score(
        score,
        description=f"Scoring on {digikey_gold.split('/')[-1]} "
        + f"against {our_gold.split('/')[-1]}.",
    )

    # Run final comparison using FN and FP
    compare_entities(
        set(score.FN),
        entity_dic=digikey_gold_dic,
        type="FN",
        gold_dic=our_gold_dic,
        outfile=outfile,
    )
    compare_entities(
        set(score.FP), type="FP", append=True, gold_dic=our_gold_dic, outfile=outfile
    )
    # NOTE: We only care about the entity_dic for FN as they are the ones
    # where we want to know what Digikey does have for manual evalutation.
    # We already know what we have (as that was the FN that Digikey missed)
    # so all we care about is what Digikey actually does have for a doc.
