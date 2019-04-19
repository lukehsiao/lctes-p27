import codecs
import csv
import logging
import os
import pdb
from collections import namedtuple

from fonduer.utils.data_model_utils import get_neighbor_cell_ngrams, get_row_ngrams
from quantiphy import Quantity

logger = logging.getLogger(__name__)

Score = namedtuple("Score", ["f1", "prec", "rec", "TP", "FP", "FN"])
LIMIT = 8


def print_scores(relation, best_result, best_b):
    logger.warning("===================================================")
    logger.warning(f"Entity-Level Gold Data score for {relation}, b={best_b:.3f}")
    logger.warning("===================================================")
    logger.warning(f"Corpus Precision {best_result.prec:.3f}")
    logger.warning(f"Corpus Recall    {best_result.rec:.3f}")
    logger.warning(f"Corpus F1        {best_result.f1:.3f}")
    logger.warning("---------------------------------------------------")
    logger.warning(
        f"TP: {len(best_result.TP)} "
        f"| FP: {len(best_result.FP)} "
        f"| FN: {len(best_result.FN)}"
    )
    logger.warning("===================================================\n")


def gold_set_to_dic(gold_set):
    gold_dic = {}
    for (doc, val) in gold_set:
        if doc in gold_dic:
            gold_dic[doc].append(val)
        else:
            gold_dic[doc] = [val]
    return gold_dic


def gold_dic_to_set(gold_dic):
    gold_set = set()
    try:
        for doc in gold_dic:
            for attr in gold_dic[doc]:
                gold_set.add((doc, attr))
    except Exception as e:
        logger.error(f"{e} while converting a {len(gold_dic)} long dict to entity set.")
        pdb.set_trace()


def get_gold_set(gold=None, docs=None, is_gain=True):

    dirname = os.path.dirname(__file__)
    gold_set = set()
    temp_dict = {}

    if gold is None:
        gold = [
            os.path.join(dirname, "data/dev/dev_gold.csv"),
            os.path.join(dirname, "data/test/test_gold.csv"),
        ]

    for filename in gold:
        with codecs.open(filename, encoding="utf-8") as csvfile:
            gold_reader = csv.reader(csvfile)
            for row in gold_reader:
                (doc, _, part, attr, val) = row
                if doc not in temp_dict:
                    temp_dict[doc] = {"typ_gbp": set(), "typ_supply_current": set()}
                if docs is None or doc.upper() in docs:
                    if attr in ["typ_gbp", "typ_supply_current"]:
                        # Allow the double of a +/- value to be valid also.
                        if val.startswith("±"):
                            (value, unit) = val.split(" ")
                            temp_dict[doc][attr].add(
                                f"{str(2 * float(value[1:]))} {unit}"
                            )
                            temp_dict[doc][attr].add(val[1:])
                        else:
                            temp_dict[doc][attr].add(val)

    # Iterate over the now-populated temp_dict to generate all tuples.
    # We use Quantities to make normalization easier during scoring.

    # NOTE: We are making the perhaps too broad assumptiion that all pairs of
    # true supply currents and GBP in a datasheet are valid.
    for doc, values in temp_dict.items():
        if is_gain:
            for gain in values["typ_gbp"]:
                gold_set.add((doc.upper(), Quantity(gain)))
        else:
            for current in values["typ_supply_current"]:
                gold_set.add((doc.upper(), Quantity(current.replace("u", "μ"))))

    return gold_set


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)

    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def cand_to_entity(c, is_gain=True):
    try:
        doc = c[0].context.sentence.document.name.upper()
    except Exception as e:
        logger.warning(f"{e}, skipping {c}")
        return

    row_ngrams = set(get_row_ngrams(c[0], n_max=1, lower=False))
    right_ngrams = set(
        [
            x[0]
            for x in get_neighbor_cell_ngrams(
                c[0], n_max=1, dist=5, directions=True, lower=False
            )
            if x[-1] == "RIGHT"
        ]
    )
    if is_gain:
        gain = c[0].context.get_span()
        # Get a set of the hertz units
        right_ngrams = set([_ for _ in right_ngrams if _ in ["kHz", "MHz", "GHz"]])
        row_ngrams = set([_ for _ in row_ngrams if _ in ["kHz", "MHz", "GHz"]])

        # Use both as a heirarchy to be more accepting of related units. Using
        # right_ngrams alone hurts recall.
        #
        # Convert to the appropriate quantities for scoring
        if len(right_ngrams) == 1:
            gain_unit = right_ngrams.pop()
        elif len(row_ngrams) == 1:
            gain_unit = row_ngrams.pop()
        else:
            # Try looking at increasingly more rows up to the LIMIT for a valid
            # current unit.
            gain_unit = None
            for i in range(LIMIT):
                rel_ngrams = set(
                    get_row_ngrams(c[0], n_max=1, spread=[-i, i], lower=False)
                )
                rel_ngrams = set([_ for _ in rel_ngrams if _ in ["kHz", "MHz", "GHz"]])
                if len(rel_ngrams) == 1:
                    gain_unit = rel_ngrams.pop()
                    break
            if not gain_unit:
                return

        try:
            result = (doc, Quantity(f"{gain} {gain_unit}"))
            yield result
        except Exception:
            logger.debug(f"{doc}: {gain} {gain_unit} is not valid.")
            return
    else:
        current = c[0].context.get_span()
        valid_units = ["nA", "mA", "μA", "uA", "µA", "\uf06dA"]

        # Get a set of the current units
        right_ngrams = set([_ for _ in right_ngrams if _ in valid_units])
        row_ngrams = set([_ for _ in row_ngrams if _ in valid_units])

        # The .replace() is needed to handle Adobe's poor conversion of unicode
        # mu.
        if len(right_ngrams) == 1:
            current_unit = right_ngrams.pop().replace("\uf06d", "μ")
        elif len(row_ngrams) == 1:
            current_unit = row_ngrams.pop().replace("\uf06d", "μ")
        else:
            # Try looking at increasingly more rows up to the LIMIT for a valid
            # current unit.
            current_unit = None
            for i in range(LIMIT):
                rel_ngrams = set(
                    get_row_ngrams(c[0], n_max=1, spread=[-i, i], lower=False)
                )
                rel_ngrams = set([_ for _ in rel_ngrams if _ in valid_units])
                if len(rel_ngrams) == 1:
                    current_unit = rel_ngrams.pop().replace("\uf06d", "μ")
                    break

            if not current_unit:
                return

        # Allow the double of a +/- value to be valid also.
        try:
            if current.startswith("±"):
                result = (
                    doc,
                    Quantity(f"{str(2 * float(current[1:]))} {current_unit}"),
                )
                yield result

                result = (doc, Quantity(f"{current[1:]} {current_unit}"))
                yield result
            else:
                result = (doc, Quantity(f"{current} {current_unit}"))
                yield result
        except Exception:
            logger.debug(f"{doc}: {current} {current_unit} is not valid.")
            return


def entity_level_scores(entities, metric=None, docs=None, corpus=None, is_gain=True):
    """Checks entity-level recall of candidates compared to gold."""
    if metric is None:
        if docs is None or len(docs) == 0:
            docs = [(doc.name).upper() for doc in corpus] if corpus else None
        metric = get_gold_set(docs=docs, is_gain=is_gain)
    if len(metric) == 0:
        logger.warn("Gold metric set is empty.")

    (TP_set, FP_set, FN_set) = entity_confusion_matrix(entities, metric)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    return Score(
        f1, prec, rec, sorted(list(TP_set)), sorted(list(FP_set)), sorted(list(FN_set))
    )


def candidates_to_entities(candidates, is_gain=True):
    # Turn CandidateSet into set of tuples
    entities = set()
    for c in candidates:
        for entity in cand_to_entity(c, is_gain=is_gain):
            entities.add(entity)
    return entities


def entity_to_candidates(entity, candidate_subset, is_gain=True):
    matches = []
    for c in candidate_subset:
        for c_entity in cand_to_entity(c, is_gain=is_gain):
            if c_entity == entity:
                matches.append(c)
    return matches


# Adaptation of existing logging format from `transistor_utils.py` but this
# time we don't look at part (just filename and val)
def compare_entities(
    entities,
    type,
    is_gain=None,
    entity_dic=None,
    gold_dic=None,
    outfile="../our_discrepancies.csv",
    append=False,
):
    """Compare given entities (filename, val) to gold labels
    and write any discrepancies to a CSV file."""

    if type is None or type not in ["FN", "FP"]:
        logger.error(f"Invalid type when writing comparison: {type}")
        return

    if gold_dic is None and is_gain is None and type != "FN":
        logger.error("Compare entities needs an attribute or gold_dic.")
        return
    elif gold_dic is None and is_gain is not None:
        gold_dic = gold_set_to_dic(get_gold_set(is_gain=is_gain))

    if entity_dic is None:
        # TODO: Right now we just convert entities to gold_dic
        # for referencing to fill `Notes:` --> But that is the same thing
        # as referencing the gold_dic.
        entity_dic = gold_set_to_dic(entities)
        # NOTE: We only care about the entity_dic for FN as they are the ones
        # where we want to know what Digikey does have for manual evalutation.
        # We already know what we have (as that was the FN that Digikey missed)
        # so all we care about is what Digikey actually does have for a doc.

    # Write discrepancies to a CSV file
    # for manual debugging
    outfile = os.path.join(os.path.dirname(__name__), outfile)
    with open(outfile, "a") if append else open(outfile, "w") as out:
        writer = csv.writer(out)
        if not append:  # Only write header row if none already exists
            writer.writerow(
                (
                    "Type:",
                    "Filename:",
                    "Our Vals:",
                    "Notes:",
                    "Discrepancy Type:",
                    "Discrepancy Notes:",
                    "Annotator:",
                )
            )
        if type == "FN":  # We only care about the entities data for `Notes:`
            for (doc, val) in entities:
                if doc.upper() in entity_dic:
                    writer.writerow(
                        (type, doc, val, f"Entity vals: {entity_dic[doc.upper()]}")
                    )
                else:
                    writer.writerow(
                        (type, doc, val, f"Entities do not have doc {doc}.")
                    )
        elif type == "FP":  # We only care about the gold data for `Notes:`
            for (doc, val) in entities:
                if doc.upper() in gold_dic:
                    writer.writerow(
                        (type, doc, val, f"Gold vals: {gold_dic[doc.upper()]}")
                    )
                else:
                    writer.writerow((type, doc, val, f"Gold does not have doc {doc}."))
