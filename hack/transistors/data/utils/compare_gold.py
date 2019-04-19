"""
This should read in two gold files, score them against eachother,
and write all discrepancies to an output CSV.
"""

import logging
import os

from hack.transistors.analysis import Relation, print_score
from hack.transistors.transistor_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Compare our gold with Digikey's gold for the analysis set of 66 docs
    # (docs that both we and Digikey have gold labels for)
    # NOTE: We use our gold as gold for this comparison against Digikey
    attribute = Relation.CE_V_MAX.value
    dirname = os.path.dirname(__name__)
    outfile = os.path.join(dirname, "../analysis/gold_discrepancies.csv")

    # Us
    our_gold = os.path.join(dirname, "../analysis/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], attribute=attribute)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Digikey
    digikey_gold = os.path.join(dirname, "../analysis/digikey_gold.csv")
    digikey_gold_set = get_gold_set(gold=[digikey_gold], attribute=attribute)
    digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

    # Score Digikey using our gold as metric
    score = entity_level_scores(
        digikey_gold_set, metric=our_gold_set, attribute=attribute
    )
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
        set(score.FP), type="FP", append=True, gold_dic=our_gold_dic, outfile=outfile
    )
