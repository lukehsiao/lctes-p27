import csv
import logging
import os
import pickle
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.learning import SparseLogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal.label_model import LabelModel
from tqdm import tqdm

from hack.opamps.opamp_lfs import FALSE, TRUE, current_lfs, gain_lfs
from hack.opamps.opamp_matchers import get_gain_matcher, get_supply_current_matcher
from hack.opamps.opamp_spaces import MentionNgramsCurrent
from hack.opamps.opamp_utils import (
    Score,
    cand_to_entity,
    candidates_to_entities,
    entity_level_scores,
    get_gold_set,
    print_scores,
)
from hack.utils import parse_dataset

# Configure logging for Hack
logger = logging.getLogger(__name__)


def dump_candidates(cands, Y_prob, outfile, is_gain=True):
    """Output the candidates and their probabilities for later analysis."""
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, outfile), "w") as csvfile:
        writer = csv.writer(csvfile)
        for i, c in enumerate(tqdm(cands)):
            for (doc, val) in cand_to_entity(c, is_gain=is_gain):
                if is_gain:
                    writer.writerow([doc, val.real / 1e3, Y_prob[i][TRUE - 1]])
                else:
                    writer.writerow([doc, val.real * 1e6, Y_prob[i][TRUE - 1]])


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info(f"Training generative model for...")
    model.train_model(L_train, n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train)
    plt.hist(marginals[:, TRUE - 1], bins=20)
    plt.savefig(os.path.join(os.path.dirname(__file__), f"opamps_marginals.pdf"))
    return marginals


def discriminative_model(
    train_cands,
    F_train,
    marginals,
    X_dev=None,
    Y_dev=None,
    n_epochs=50,
    lr=0.001,
    gpu=None,
):
    disc_model = SparseLogisticRegression()

    logger.info("Training discriminative model...")
    marginals = []
    for y in Y_dev:
        if y == 1:
            marginals.append([1.0, 0.0])
        else:
            marginals.append([0.0, 1.0])
    marginals = np.array(marginals)
    if gpu:
        disc_model.train(
            X_dev,
            marginals,
            X_dev=X_dev,
            Y_dev=Y_dev,
            n_epochs=n_epochs,
            lr=lr,
            host_device="GPU",
        )
    else:
        disc_model.train(
            X_dev,
            marginals,
            X_dev=X_dev,
            Y_dev=Y_dev,
            n_epochs=n_epochs,
            lr=lr,
            host_device="CPU",
        )

    logger.info("Done.")

    return disc_model


def output_csv(cands, Y_prob, is_gain=True, append=False):
    dirname = os.path.dirname(__file__)
    if is_gain:
        filename = "output_gain.csv"
    else:
        filename = "output_current.csv"
    filename = os.path.join(dirname, filename)

    if append:
        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile)
            for i, c in enumerate(tqdm(cands)):
                for entity in cand_to_entity(c, is_gain=is_gain):
                    if is_gain:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real / 1e3,
                                c[0].context.sentence.position,
                                Y_prob[i][TRUE - 1],
                            ]
                        )
                    else:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real * 1e6,
                                c[0].context.sentence.position,
                                Y_prob[i][TRUE - 1],
                            ]
                        )
    else:
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            if is_gain:
                writer.writerow(["Document", "GBWP (kHz)", "sent", "p"])
            else:
                writer.writerow(["Document", "Supply Current (uA)", "sent", "p"])

            for i, c in enumerate(tqdm(cands)):
                for entity in cand_to_entity(c, is_gain=is_gain):
                    if is_gain:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real / 1e3,
                                c[0].context.sentence.position,
                                Y_prob[i][TRUE - 1],
                            ]
                        )
                    else:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real * 1e6,
                                c[0].context.sentence.position,
                                Y_prob[i][TRUE - 1],
                            ]
                        )


def scoring(disc_model, cands, docs, F_mat, is_gain=True, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Iterate over a range of `b` values in order to find the b with the
    # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
    Y_prob = disc_model.marginals((cands, F_mat))
    logger.info("Grab Y_prob.")

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            test_score = np.array(
                [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
            )
            true_pred = [cands[_] for _ in np.nditer(np.where(test_score == TRUE))]
        except Exception as e:
            logger.debug(f"{e}, skipping.")
            break
        result = entity_level_scores(
            candidates_to_entities(true_pred, is_gain=is_gain),
            corpus=docs,
            is_gain=is_gain,
        )
        logger.info(
            f"({b:.3f}), f1:{result.f1:.3f} p:{result.prec:.3f} r:{result.rec:.3f}"
        )

        if result.f1 > best_result.f1:
            best_result = result
            best_b = b

    return best_result, best_b


def main(
    conn_string,
    gain=False,
    current=False,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    re_label=False,
    gpu=None,
    parallel=8,
    log_dir="logs",
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
    if gain:
        rel_list.append("gain")

    if current:
        rel_list.append("current")

    logger.info(f"=" * 30)
    logger.info(f"Running with parallel: {parallel}, max_docs: {max_docs}")

    session = Meta.init(conn_string).Session()

    # Parsing
    start = timer()
    logger.info(f"Starting parsing...")
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=parse, parallel=parallel, max_docs=max_docs
    )
    logger.debug(f"Done")
    end = timer()
    logger.warning(f"Parse Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"# of Documents: {len(docs)}")
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
    if gain:
        Gain = mention_subclass("Gain")
        gain_matcher = get_gain_matcher()
        gain_ngrams = MentionNgrams(n_max=2)
        mentions.append(Gain)
        ngrams.append(gain_ngrams)
        matchers.append(gain_matcher)

    if current:
        Current = mention_subclass("SupplyCurrent")
        current_matcher = get_supply_current_matcher()
        current_ngrams = MentionNgramsCurrent(n_max=3)
        mentions.append(Current)
        ngrams.append(current_ngrams)
        matchers.append(current_matcher)

    mention_extractor = MentionExtractor(session, mentions, ngrams, matchers)

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")

    if gain:
        logger.info(f"Total Gain: {session.query(Gain).count()}")

    if current:
        logger.info(f"Total Current: {session.query(Current).count()}")

    cand_classes = []
    if gain:
        GainCand = candidate_subclass("GainCand", [Gain])
        cand_classes.append(GainCand)
    if current:
        CurrentCand = candidate_subclass("CurrentCand", [Current])
        cand_classes.append(CurrentCand)

    candidate_extractor = CandidateExtractor(session, cand_classes)

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)
    logger.info(f"Total train candidate: {len(train_cands[0]) + len(train_cands[1])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0]) + len(dev_cands[1])}")
    logger.info(f"Total test candidate: {len(test_cands[0]) + len(test_cands[1])}")

    logger.info("Done w/ candidate extraction.")
    end = timer()
    logger.warning(f"CE Time (min): {((end - start) / 60.0):.1f}")

    # First, check total recall
    #  result = entity_level_scores(dev_cands[0], corpus=dev_docs)
    #  logger.info(f"Gain Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(test_cands[0], corpus=test_docs)
    #  logger.info(f"Gain Total Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #
    #  result = entity_level_scores(dev_cands[1], corpus=dev_docs, is_gain=False)
    #  logger.info(f"Current Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(test_cands[1], corpus=test_docs, is_gain=False)
    #  logger.info(f"Current Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")

    start = timer()
    featurizer = Featurizer(session, cand_classes)

    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    # Serialize feature matrices on first run
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

    start = timer()
    logger.info("Labeling training data...")
    labeler = Labeler(session, cand_classes)
    lfs = []
    if gain:
        lfs.append(gain_lfs)

    if current:
        lfs.append(current_lfs)

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=0, lfs=lfs, train=True, parallelism=parallel)
    elif re_label:
        logger.info("Re-applying LFs...")
        labeler.update(split=0, lfs=lfs, parallelism=parallel)

    logger.info("Done...")

    logger.info("Getting label matrices...")
    L_train = labeler.get_label_matrices(train_cands)
    logger.info("Done...")

    end = timer()
    logger.warning(f"Weak Supervision Time (min): {((end - start) / 60.0):.1f}")

    if gain:
        relation = "gain"
        idx = rel_list.index(relation)

        logger.info("Score Gain.")
        dev_gold_entities = get_gold_set(is_gain=True)
        L_dev_gt = []
        for c in dev_cands[idx]:
            flag = FALSE
            for entity in cand_to_entity(c, is_gain=True):
                if entity in dev_gold_entities:
                    flag = TRUE
            L_dev_gt.append(flag)

        marginals = generative_model(L_train[idx])
        disc_models = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals,
            X_dev=(dev_cands[idx], F_dev[idx]),
            Y_dev=L_dev_gt,
            n_epochs=500,
            gpu=gpu,
        )
        best_result, best_b = scoring(
            disc_models, test_cands[idx], test_docs, F_test[idx], num=50
        )

        print_scores(relation, best_result, best_b)

        logger.info("Output CSV files for Opo and Digi-key Analysis.")
        Y_prob = disc_models.marginals((train_cands[idx], F_train[idx]))
        output_csv(train_cands[idx], Y_prob, is_gain=True)

        Y_prob = disc_models.marginals((test_cands[idx], F_test[idx]))
        output_csv(test_cands[idx], Y_prob, is_gain=True, append=True)
        dump_candidates(test_cands[idx], Y_prob, "gain_test_probs.csv", is_gain=True)

        Y_prob = disc_models.marginals((dev_cands[idx], F_dev[idx]))
        output_csv(dev_cands[idx], Y_prob, is_gain=True, append=True)
        dump_candidates(dev_cands[idx], Y_prob, "gain_dev_probs.csv", is_gain=True)

    if current:
        relation = "current"
        idx = rel_list.index(relation)

        logger.info("Score Current.")
        dev_gold_entities = get_gold_set(is_gain=False)
        L_dev_gt = []
        for c in dev_cands[idx]:
            flag = FALSE
            for entity in cand_to_entity(c, is_gain=False):
                if entity in dev_gold_entities:
                    flag = TRUE
            L_dev_gt.append(flag)

        marginals = generative_model(L_train[idx])

        disc_models = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals,
            X_dev=(dev_cands[idx], F_dev[idx]),
            Y_dev=L_dev_gt,
            n_epochs=100,
            gpu=gpu,
        )
        best_result, best_b = scoring(
            disc_models, test_cands[idx], test_docs, F_test[idx], is_gain=False, num=50
        )

        print_scores(relation, best_result, best_b)

        logger.info("Output CSV files for Opo and Digi-key Analysis.")
        # Dump CSV files for digi-key analysis and Opo comparison
        Y_prob = disc_models.marginals((train_cands[idx], F_train[idx]))
        output_csv(train_cands[idx], Y_prob, is_gain=False)

        Y_prob = disc_models.marginals((test_cands[idx], F_test[idx]))
        output_csv(test_cands[idx], Y_prob, is_gain=False, append=True)
        dump_candidates(
            test_cands[idx], Y_prob, "current_test_probs.csv", is_gain=False
        )

        Y_prob = disc_models.marginals((dev_cands[idx], F_dev[idx]))
        output_csv(dev_cands[idx], Y_prob, is_gain=False, append=True)
        dump_candidates(dev_cands[idx], Y_prob, "current_dev_probs.csv", is_gain=False)

    end = timer()
    logger.warning(
        f"Classification AND dump data Time (min): {((end - start) / 60.0):.1f}"
    )
