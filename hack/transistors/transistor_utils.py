import codecs
import csv
import logging
import os
import pdb
from builtins import range
from collections import namedtuple

from fonduer.supervision.models import GoldLabel, GoldLabelKey

from hack.transistors.transistor_lfs import FALSE, TRUE

try:
    from IPython import get_ipython

    if "IPKernelApp" not in get_ipython().config:
        raise ImportError("console")
except (AttributeError, ImportError):
    from tqdm import tqdm
else:
    from tqdm import tqdm_notebook as tqdm


logger = logging.getLogger(__name__)

Score = namedtuple("Score", ["f1", "prec", "rec", "TP", "FP", "FN"])


def gold_set_to_dic(gold_set):
    gold_dic = {}
    for (doc, part, val) in gold_set:
        if doc in gold_dic:
            if part in gold_dic[doc]:
                gold_dic[doc][part].append(val)
            else:
                gold_dic[doc][part] = [val]
        else:
            gold_dic[doc] = {part: [val]}
    return gold_dic


def gold_dic_to_set(gold_dic):
    gold_set = set()
    try:
        for doc in gold_dic:
            for part in gold_dic[doc]:
                for attr in gold_dic[doc][part]:
                    gold_set.add((doc, part, attr))
    except Exception as e:
        logger.error(f"{e} while converting a {len(gold_dic)} long dict to entity set.")
        pdb.set_trace()


def get_gold_set(
    doc_on=True, part_on=True, val_on=True, attribute=None, docs=None, gold=None
):
    if gold is None:
        dirname = os.path.dirname(__file__)
        gold = [
            os.path.join(dirname, "data/dev/dev_gold.csv"),
            os.path.join(dirname, "data/test/test_gold.csv"),
        ]

    gold_set = set()
    for filename in gold:
        with codecs.open(filename, encoding="utf-8") as csvfile:
            gold_reader = csv.reader(csvfile)
            for row in gold_reader:
                (doc, _, part, attr, val) = row
                if docs is None or doc.upper() in docs:
                    if attribute and attr != attribute:
                        continue
                    else:
                        key = []
                        if doc_on:
                            key.append(doc.upper())
                        if part_on:
                            key.append(part.upper())
                        if val_on:
                            key.append(val.upper())
                        gold_set.add(tuple(key))

    return gold_set


def load_transistor_labels(session, candidate_classes, attribs, annotator_name="gold"):
    """Bulk insert hardware GoldLabels.

    :param session: The database session to use.
    :param candidate_classes: Which candidate_classes to load labels for.
    :param attribs: Which attributes to load labels for (e.g. "stg_temp_max").
    """
    # Check that candidate_classes is iterable
    candidate_classes = (
        candidate_classes
        if isinstance(candidate_classes, (list, tuple))
        else [candidate_classes]
    )

    # Check that attribs is iterable
    attribs = attribs if isinstance(attribs, (list, tuple)) else [attribs]

    if len(candidate_classes) != len(attribs):
        logger.warning("candidate_classes and attribs must be the same length.")
        return

    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()

    # Add the gold key
    if ak is None:
        ak = GoldLabelKey(
            name=annotator_name,
            candidate_classes=[_.__tablename__ for _ in candidate_classes],
        )
        session.add(ak)
        session.commit()

    # Bulk insert candidate labels
    candidates = []
    for candidate_class, attrib in zip(candidate_classes, attribs):
        candidates.extend(session.query(candidate_class).all())

        logger.info(f"Loading {attrib} for {candidate_class.__tablename__}...")
        gold_set = get_gold_set(attribute=attrib)

        cand_total = len(candidates)
        logger.info(f"Loading {cand_total} candidate labels...")
        labels = 0

        cands = []
        values = []
        for i, c in enumerate(tqdm(candidates)):
            doc = (c[0].context.sentence.document.name).upper()
            part = (c[0].context.get_span()).upper()
            val = ("".join(c[1].context.get_span().split())).upper()

            label = session.query(GoldLabel).filter(GoldLabel.candidate == c).first()
            if label is None:
                if (doc, part, val) in gold_set:
                    values.append(TRUE)
                else:
                    values.append(FALSE)

                cands.append(c)
                labels += 1

        # Only insert the labels which were not already present
        session.bulk_insert_mappings(
            GoldLabel,
            [
                {"candidate_id": cand.id, "keys": [annotator_name], "values": [val]}
                for (cand, val) in zip(cands, values)
            ],
        )
        session.commit()

        logger.info(f"GoldLabels created: {labels}")


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def compare_entities(
    entities,
    attribute=None,
    entity_dic=None,
    gold_dic=None,
    type=None,
    outfile="analysis_discrepancies.csv",
    append=False,
):
    """Compare given entities to gold labels
    and write any discrepancies to a CSV file."""

    if type is None or type not in ["FN", "FP"]:
        logger.error(f"Invalid discrepancy type {type}")
        pdb.set_trace()
    if gold_dic is None and attribute is None:
        logger.error("Compare entities needs an attribute or gold_dic.")
        pdb.set_trace()
    elif gold_dic is None and attribute is not None and type == "FP":
        gold_dic = gold_set_to_dic(get_gold_set(attribute=attribute))

    if entity_dic is None:
        # TODO: Right now we just convert entities to gold_dic
        # for referencing to fill `Notes:` --> But that is the same thing
        # as referencing the gold_dic.
        entity_dic = gold_set_to_dic(entities)

    # Write discrepancies to a CSV file
    # for manual debugging
    outfile = os.path.join(os.path.dirname(__name__), outfile)
    with open(outfile, "a") if append else open(outfile, "w") as out:
        writer = csv.writer(out, lineterminator="\n")
        if not append:  # Only write header row if none already exists
            writer.writerow(
                (
                    "Type:",
                    "Filename:",
                    "Part:",
                    "Our Vals:",
                    "Notes:",
                    "Discrepancy Type:",
                    "Discrepancy Notes:",
                    "Annotator:",
                )
            )
        if type == "FN":  # We only care about the entities data for `Notes:`
            for (doc, part, val) in entities:
                if doc.upper() in entity_dic:
                    if part.upper() in entity_dic[doc.upper()]:
                        writer.writerow(
                            (
                                type,
                                doc,
                                part,
                                val,
                                f"Entity vals: {entity_dic[doc.upper()][part.upper()]}",
                                "Missing value.",
                                "",
                                "Bot",
                            )
                        )
                    else:
                        writer.writerow(
                            (
                                type,
                                doc,
                                part,
                                val,
                                f"Entity parts: {entity_dic[doc]}",
                                "Missing part.",
                                "",
                                "Bot",
                            )
                        )
                else:
                    writer.writerow(
                        (
                            type,
                            doc,
                            part,
                            val,
                            f"Entities do not have doc {doc}.",
                            "Missing doc.",
                            "",
                            "Bot",
                        )
                    )
        elif type == "FP":  # We only care about the gold data for `Notes:`
            for (doc, part, val) in entities:
                if doc.upper() in gold_dic:
                    if part.upper() in gold_dic[doc.upper()]:
                        writer.writerow(
                            (
                                type,
                                doc,
                                part,
                                val,
                                f"Gold vals: {gold_dic[doc.upper()][part.upper()]}",
                                "Invalid value.",
                                "",
                                "Bot",
                            )
                        )
                    else:
                        writer.writerow(
                            (
                                type,
                                doc,
                                part,
                                val,
                                f"Gold parts: {gold_dic[doc]}",
                                "Invalid part.",
                                "",
                                "Bot",
                            )
                        )
                else:
                    writer.writerow(
                        (
                            type,
                            doc,
                            part,
                            val,
                            f"Gold does not have doc {doc}.",
                            "Gold is missing doc.",
                            "",
                            "Bot",
                        )
                    )


def entity_level_scores(entities, attribute=None, corpus=None, metric=None, docs=None):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        entity_level_total_recall(candidates, 'stg_temp_min')
    """
    if docs is None or len(docs) == 0:
        docs = [(doc.name).upper() for doc in corpus] if corpus else None
    val_on = attribute is not None
    if metric is None:
        metric = get_gold_set(
            docs=docs, doc_on=True, part_on=True, val_on=val_on, attribute=attribute
        )
    if len(metric) == 0:
        logger.info(f"Attribute: {attribute}")
        logger.error("Gold metric set is empty.")

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


def get_implied_parts(part, doc, parts_by_doc):
    if parts_by_doc:
        for p in parts_by_doc[doc]:
            if p.startswith(part) and len(part) >= 4:
                yield p


def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple(
            [c[0].context.sentence.document.name.upper()]
            + [c[i].context.get_span().upper() for i in range(len(c))]
        )
        c_entity = tuple([str(x) for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches


def cand_to_entity(c):
    """Convert a single candidate to an entity."""
    part = c[0].context.get_span()
    doc = c[0].context.sentence.document.name.upper()
    val = c[1].context.get_span()
    return (doc, part, val)


def candidates_to_entities(
    candidates, val_on=True, parts_by_doc=None, progress_bar=True
):
    # Turn CandidateSet into set of tuples
    entities = set()
    if progress_bar:
        candidates = tqdm(candidates)
    for i, c in enumerate(candidates):
        part = c[0].context.get_span()
        doc = c[0].context.sentence.document.name.upper()
        if val_on:
            val = c[1].context.get_span()
        for p in get_implied_parts(part, doc, parts_by_doc):
            if val_on:
                entities.add((doc, p, val))
            else:
                entities.add((doc, p))
    return entities


def _files_in_dir(path):
    """Return the filenames, but drops the ".pdf" extension."""
    return [f[:-4] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
