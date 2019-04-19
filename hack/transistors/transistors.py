#!/usr/bin/env python
# coding: utf-8

import csv
import logging
import os
import pickle
from timeit import default_timer as timer

import numpy as np
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.features import Featurizer
from fonduer.learning import SparseLogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal.label_model import LabelModel

from hack.transistors.transistor_lfs import (
    TRUE,
    ce_v_max_lfs,
    polarity_lfs,
    stg_temp_max_lfs,
    stg_temp_min_lfs,
)
from hack.transistors.transistor_matchers import get_matcher
from hack.transistors.transistor_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from hack.transistors.transistor_throttlers import (
    ce_v_max_filter,
    polarity_filter,
    stg_temp_filter,
)
from hack.transistors.transistor_utils import (
    Score,
    cand_to_entity,
    candidates_to_entities,
    entity_level_scores,
    load_transistor_labels,
)
from hack.utils import parse_dataset

logger = logging.getLogger(__name__)


def load_labels(session, relation, cand, first_time=True):
    if first_time:
        logger.info(f"Loading gold labels for {relation}")
        load_transistor_labels(session, [cand], [relation], annotator_name="gold")


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info("Training generative model...")
    model.train_model(L_train, n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train)
    return marginals


def discriminative_model(
    train_cands, F_train, marginals, n_epochs=50, lr=0.001, gpu=None
):
    disc_model = SparseLogisticRegression()

    logger.info("Training discriminative model...")
    if gpu:
        disc_model.train(
            (train_cands, F_train),
            marginals,
            n_epochs=n_epochs,
            lr=lr,
            host_device="GPU",
        )
    else:
        disc_model.train(
            (train_cands, F_train),
            marginals,
            n_epochs=n_epochs,
            lr=lr,
            host_device="CPU",
        )

    logger.info("Done.")

    return disc_model


def scoring(relation, disc_model, test_cands, test_docs, F_test, parts_by_doc, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Iterate over a range of `b` values in order to find the b with the
    # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
    Y_prob = disc_model.marginals((test_cands, F_test))

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            test_score = np.array(
                [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
            )
            true_pred = [test_cands[_] for _ in np.nditer(np.where(test_score == TRUE))]
            result = entity_level_scores(
                candidates_to_entities(
                    true_pred, parts_by_doc=parts_by_doc, progress_bar=False
                ),
                attribute=relation,
                corpus=test_docs,
            )
            logger.info(
                f"b:{b:.3f} f1:{result.f1:.3f} p:{result.prec:.3f} r:{result.rec:.3f}"
            )
            if result.f1 > best_result.f1:
                best_result = result
                best_b = b
        except Exception as e:
            logger.debug(f"{e}, skipping.")
            break

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
    return best_result, best_b


def dump_candidates(cands, Y_prob, outfile):
    """Output the ce_v_max candidates and their probabilities for later analysis."""
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, outfile), "w") as csvfile:
        writer = csv.writer(csvfile)
        for i, c in enumerate(cands):
            (doc, part, val) = cand_to_entity(c)
            writer.writerow([doc, part, val, Y_prob[i][TRUE - 1]])


def main(
    conn_string,
    stg_temp_min=False,
    stg_temp_max=False,
    polarity=False,
    ce_v_max=False,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    re_label=False,
    gpu=None,
    parallel=4,
    log_dir=None,
    verbose=False,
):
    # Setup initial configuration
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if not log_dir:
        log_dir = "logs"

    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    dirname = os.path.dirname(os.path.abspath(__file__))
    init_logging(log_dir=os.path.join(dirname, log_dir), level=level)

    rel_list = []
    if stg_temp_min:
        rel_list.append("stg_temp_min")

    if stg_temp_max:
        rel_list.append("stg_temp_max")

    if polarity:
        rel_list.append("polarity")

    if ce_v_max:
        rel_list.append("ce_v_max")

    session = Meta.init(conn_string).Session()

    # Parsing
    logger.info(f"Starting parsing...")
    start = timer()
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=parse, parallel=parallel, max_docs=max_docs
    )
    end = timer()
    logger.warning(f"Parse Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"# of train Documents: {len(train_docs)}")
    logger.info(f"# of dev Documents: {len(dev_docs)}")
    logger.info(f"# of test Documents: {len(test_docs)}")
    logger.info(f"Documents: {session.query(Document).count()}")
    logger.info(f"Sections: {session.query(Section).count()}")
    logger.info(f"Paragraphs: {session.query(Paragraph).count()}")
    logger.info(f"Sentences: {session.query(Sentence).count()}")
    logger.info(f"Figures: {session.query(Figure).count()}")

    # Mention Extraction
    start = timer()
    mentions = []
    ngrams = []
    matchers = []

    # Only do those that are enabled
    Part = mention_subclass("Part")
    part_matcher = get_matcher("part")
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)

    mentions.append(Part)
    ngrams.append(part_ngrams)
    matchers.append(part_matcher)

    if stg_temp_min:
        StgTempMin = mention_subclass("StgTempMin")
        stg_temp_min_matcher = get_matcher("stg_temp_min")
        stg_temp_min_ngrams = MentionNgramsTemp(n_max=2)

        mentions.append(StgTempMin)
        ngrams.append(stg_temp_min_ngrams)
        matchers.append(stg_temp_min_matcher)

    if stg_temp_max:
        StgTempMax = mention_subclass("StgTempMax")
        stg_temp_max_matcher = get_matcher("stg_temp_max")
        stg_temp_max_ngrams = MentionNgramsTemp(n_max=2)

        mentions.append(StgTempMax)
        ngrams.append(stg_temp_max_ngrams)
        matchers.append(stg_temp_max_matcher)

    if polarity:
        Polarity = mention_subclass("Polarity")
        polarity_matcher = get_matcher("polarity")
        polarity_ngrams = MentionNgrams(n_max=1)

        mentions.append(Polarity)
        ngrams.append(polarity_ngrams)
        matchers.append(polarity_matcher)

    if ce_v_max:
        CeVMax = mention_subclass("CeVMax")
        ce_v_max_matcher = get_matcher("ce_v_max")
        ce_v_max_ngrams = MentionNgramsVolt(n_max=1)

        mentions.append(CeVMax)
        ngrams.append(ce_v_max_ngrams)
        matchers.append(ce_v_max_matcher)

    mention_extractor = MentionExtractor(session, mentions, ngrams, matchers)

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")
    logger.info(f"Total Part: {session.query(Part).count()}")
    if stg_temp_min:
        logger.info(f"Total StgTempMin: {session.query(StgTempMin).count()}")
    if stg_temp_max:
        logger.info(f"Total StgTempMax: {session.query(StgTempMax).count()}")
    if polarity:
        logger.info(f"Total Polarity: {session.query(Polarity).count()}")
    if ce_v_max:
        logger.info(f"Total CeVMax: {session.query(CeVMax).count()}")

    # Candidate Extraction
    cands = []
    throttlers = []
    if stg_temp_min:
        PartStgTempMin = candidate_subclass("PartStgTempMin", [Part, StgTempMin])
        stg_temp_min_throttler = stg_temp_filter

        cands.append(PartStgTempMin)
        throttlers.append(stg_temp_min_throttler)

    if stg_temp_max:
        PartStgTempMax = candidate_subclass("PartStgTempMax", [Part, StgTempMax])
        stg_temp_max_throttler = stg_temp_filter

        cands.append(PartStgTempMax)
        throttlers.append(stg_temp_max_throttler)

    if polarity:
        PartPolarity = candidate_subclass("PartPolarity", [Part, Polarity])
        polarity_throttler = polarity_filter

        cands.append(PartPolarity)
        throttlers.append(polarity_throttler)

    if ce_v_max:
        PartCeVMax = candidate_subclass("PartCeVMax", [Part, CeVMax])
        ce_v_max_throttler = ce_v_max_filter

        cands.append(PartCeVMax)
        throttlers.append(ce_v_max_throttler)

    candidate_extractor = CandidateExtractor(session, cands, throttlers=throttlers)

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)
            num_cands = session.query(Candidate).filter(Candidate.split == i).count()
            logger.info(f"Candidates in split={i}: {num_cands}")

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)

    end = timer()
    logger.warning(f"Candidate Extraction Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"Total train candidate: {sum(len(_) for _ in train_cands)}")
    logger.info(f"Total dev candidate: {sum(len(_) for _ in dev_cands)}")
    logger.info(f"Total test candidate: {sum(len(_) for _ in test_cands)}")

    pickle_file = os.path.join(dirname, "data/parts_by_doc_new.pkl")
    with open(pickle_file, "rb") as f:
        parts_by_doc = pickle.load(f)

    # Check total recall
    for i, name in enumerate(rel_list):
        logger.info(name)
        result = entity_level_scores(
            candidates_to_entities(dev_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=dev_docs,
        )
        logger.info(f"{name} Total Dev Recall: {result.rec:.3f}")
        result = entity_level_scores(
            candidates_to_entities(test_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=test_docs,
        )
        logger.info(f"{name} Total Test Recall: {result.rec:.3f}")

    # Featurization
    start = timer()
    cands = []
    if stg_temp_min:
        cands.append(PartStgTempMin)

    if stg_temp_max:
        cands.append(PartStgTempMax)

    if polarity:
        cands.append(PartPolarity)

    if ce_v_max:
        cands.append(PartCeVMax)

    featurizer = Featurizer(session, cands)
    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    if first_time:
        F_train = featurizer.get_feature_matrices(train_cands)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        F_test = featurizer.get_feature_matrices(test_cands)
        end = timer()
        logger.warning(f"Featurization Time (min): {((end - start) / 60.0):.1f}")

        pickle.dump(F_train, open(os.path.join(dirname, "F_train.pkl"), "wb"))
        pickle.dump(F_dev, open(os.path.join(dirname, "F_dev.pkl"), "wb"))
        pickle.dump(F_test, open(os.path.join(dirname, "F_test.pkl"), "wb"))
    else:
        F_train = pickle.load(open(os.path.join(dirname, "F_train.pkl"), "rb"))
        F_dev = pickle.load(open(os.path.join(dirname, "F_dev.pkl"), "rb"))
        F_test = pickle.load(open(os.path.join(dirname, "F_test.pkl"), "rb"))
    logger.info("Done.")

    for i, cand in enumerate(cands):
        logger.info(f"{cand} Train shape: {F_train[i].shape}")
        logger.info(f"{cand} Test shape: {F_test[i].shape}")
        logger.info(f"{cand} Dev shape: {F_dev[i].shape}")

    logger.info("Labeling training data...")

    # Labeling
    start = timer()
    lfs = []
    if stg_temp_min:
        lfs.append(stg_temp_min_lfs)

    if stg_temp_max:
        lfs.append(stg_temp_max_lfs)

    if polarity:
        lfs.append(polarity_lfs)

    if ce_v_max:
        lfs.append(ce_v_max_lfs)

    labeler = Labeler(session, cands)

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=0, lfs=lfs, train=True, parallelism=parallel)
        logger.info("Done...")

        # Uncomment if debugging LFs
        #  load_transistor_labels(session, cands, ["ce_v_max"])
        #  labeler.apply(split=1, lfs=lfs, train=False, parallelism=parallel)
        #  labeler.apply(split=2, lfs=lfs, train=False, parallelism=parallel)

    elif re_label:
        logger.info("Updating LFs...")
        labeler.update(split=0, lfs=lfs, parallelism=parallel)
        logger.info("Done...")

        # Uncomment if debugging LFs
        #  labeler.apply(split=1, lfs=lfs, train=False, parallelism=parallel)
        #  labeler.apply(split=2, lfs=lfs, train=False, parallelism=parallel)

    logger.info("Getting label matrices...")

    L_train = labeler.get_label_matrices(train_cands)

    # Uncomment if debugging LFs
    #  L_dev = labeler.get_label_matrices(dev_cands)
    #  L_dev_gold = labeler.get_gold_labels(dev_cands, annotator="gold")
    #
    #  L_test = labeler.get_label_matrices(test_cands)
    #  L_test_gold = labeler.get_gold_labels(test_cands, annotator="gold")

    logger.info("Done.")

    end = timer()
    logger.warning(f"Supervision Time (min): {((end - start) / 60.0):.1f}")

    start = timer()
    if stg_temp_min:
        relation = "stg_temp_min"
        idx = rel_list.index(relation)
        marginals_stg_temp_min = generative_model(L_train[idx])
        disc_model_stg_temp_min = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals_stg_temp_min,
            n_epochs=100,
            gpu=gpu,
        )
        best_result, best_b = scoring(
            relation,
            disc_model_stg_temp_min,
            test_cands[idx],
            test_docs,
            F_test[idx],
            parts_by_doc,
            num=100,
        )

    if stg_temp_max:
        relation = "stg_temp_max"
        idx = rel_list.index(relation)
        marginals_stg_temp_max = generative_model(L_train[idx])
        disc_model_stg_temp_max = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals_stg_temp_max,
            n_epochs=100,
            gpu=gpu,
        )
        best_result, best_b = scoring(
            relation,
            disc_model_stg_temp_max,
            test_cands[idx],
            test_docs,
            F_test[idx],
            parts_by_doc,
            num=100,
        )

    if polarity:
        relation = "polarity"
        idx = rel_list.index(relation)
        marginals_polarity = generative_model(L_train[idx])
        disc_model_polarity = discriminative_model(
            train_cands[idx], F_train[idx], marginals_polarity, n_epochs=100, gpu=gpu
        )
        best_result, best_b = scoring(
            relation,
            disc_model_polarity,
            test_cands[idx],
            test_docs,
            F_test[idx],
            parts_by_doc,
            num=100,
        )

    if ce_v_max:
        relation = "ce_v_max"
        idx = rel_list.index(relation)

        # Can be uncommented for use in debugging labeling functions
        #  logger.info("Updating labeling function summary...")
        #  keys = labeler.get_keys()
        #  logger.info("Summary for train set labeling functions:")
        #  df = analysis.lf_summary(L_train[idx], lf_names=keys)
        #  logger.info(f"\n{df.to_string()}")
        #
        #  logger.info("Summary for dev set labeling functions:")
        #  df = analysis.lf_summary(
        #      L_dev[idx],
        #      lf_names=keys,
        #      Y=L_dev_gold[idx].todense().reshape(-1).tolist()[0],
        #  )
        #  logger.info(f"\n{df.to_string()}")
        #
        #  logger.info("Summary for test set labeling functions:")
        #  df = analysis.lf_summary(
        #      L_test[idx],
        #      lf_names=keys,
        #      Y=L_test_gold[idx].todense().reshape(-1).tolist()[0],
        #  )
        #  logger.info(f"\n{df.to_string()}")

        marginals_ce_v_max = generative_model(L_train[idx])
        disc_model_ce_v_max = discriminative_model(
            train_cands[idx], F_train[idx], marginals_ce_v_max, n_epochs=100, gpu=gpu
        )

        # Can be uncommented to view score on development set
        #  best_result, best_b = scoring(
        #      relation,
        #      disc_model_ce_v_max,
        #      dev_cands[idx],
        #      dev_docs,
        #      F_dev[idx],
        #      parts_by_doc,
        #      num=100,
        #  )

        best_result, best_b = scoring(
            relation,
            disc_model_ce_v_max,
            test_cands[idx],
            test_docs,
            F_test[idx],
            parts_by_doc,
            num=100,
        )

    end = timer()
    logger.warning(f"Classification Time (min): {((end - start) / 60.0):.1f}")

    # Dump CSV files for CE_V_MAX for digi-key analysis
    if ce_v_max:
        relation = "ce_v_max"
        idx = rel_list.index(relation)
        Y_prob = disc_model_ce_v_max.marginals((test_cands[idx], F_test[idx]))
        dump_candidates(test_cands[idx], Y_prob, "ce_v_max_test_probs.csv")
        Y_prob = disc_model_ce_v_max.marginals((dev_cands[idx], F_dev[idx]))
        dump_candidates(dev_cands[idx], Y_prob, "ce_v_max_dev_probs.csv")

    # Dump CSV files for POLARITY for digi-key analysis
    if polarity:
        relation = "polarity"
        idx = rel_list.index(relation)
        Y_prob = disc_model_polarity.marginals((test_cands[idx], F_test[idx]))
        dump_candidates(test_cands[idx], Y_prob, "polarity_test_probs.csv")
        Y_prob = disc_model_polarity.marginals((dev_cands[idx], F_dev[idx]))
        dump_candidates(dev_cands[idx], Y_prob, "polarity_dev_probs.csv")
