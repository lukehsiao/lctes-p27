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


def main(
    relation=Relation.CE_V_MAX.value,
    outfile="analysis/ce_v_max_digikey_discrepancies.csv",
):
    # Compare our gold with Digikey's gold for the analysis set of 66 docs
    # (docs that both we and Digikey have gold labels for)
    # NOTE: We use our gold as gold for this comparison against Digikey
    dirname = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(dirname, outfile)

    # Us
    our_gold = os.path.join(dirname, "data/analysis/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], attribute=relation)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Digikey
    digikey_gold = os.path.join(dirname, "data/analysis/digikey_gold.csv")
    digikey_gold_set = get_gold_set(gold=[digikey_gold], attribute=relation)
    digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

    # Score Digikey using our gold as metric
    score = entity_level_scores(
        digikey_gold_set, metric=our_gold_set, attribute=relation
    )
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
