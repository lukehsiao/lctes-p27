import logging
import os

from fonduer.parser import Parser
from fonduer.parser.preprocessors import HTMLDocPreprocessor

logger = logging.getLogger(__name__)


def load_ids(filename):
    """Loads document ids from a newline separated list of filenames."""
    fin = open(filename, "r")
    return set(_.strip() for _ in fin)


def _files_in_dir(path):
    """Return the filenames, but drops the ".pdf" extension."""
    return [f[:-4] for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_dataset(
    session, dirname, first_time=False, max_docs=float("inf"), parallel=4
):
    """Parse the dataset into dev, test, and train.

    This expects that the data is located in data/dev/, data/test/, data/train/
    directories, and each of those contains html/ and pdf/. Also expects that
    the filenames of the HTML and PDF match.

    :param session: The database session
    :param max_docs: The maximum number of documents to parse from each set.
        Defaults to parsing all documents.
    :rtype: (all_docs, train_docs, dev_docs, test_docs)
    """
    train_docs = set()
    dev_docs = set()
    test_docs = set()

    if first_time:
        for division in ["dev", "test", "train"]:
            logger.info(f"Parsing {division}...")
            html_path = os.path.join(dirname, f"data/{division}/html/")
            pdf_path = os.path.join(dirname, f"data/{division}/pdf/")

            doc_preprocessor = HTMLDocPreprocessor(html_path, max_docs=max_docs)

            corpus_parser = Parser(
                session, structural=True, lingual=True, visual=True, pdf_path=pdf_path
            )
            # Do not want to clear the database when parsing test and train
            if division == "dev":
                corpus_parser.apply(doc_preprocessor, parallelism=parallel)
                dev_docs = set(corpus_parser.get_last_documents())
            if division == "test":
                corpus_parser.apply(doc_preprocessor, parallelism=parallel, clear=False)
                test_docs = set(corpus_parser.get_last_documents())
            if division == "train":
                corpus_parser.apply(doc_preprocessor, parallelism=parallel, clear=False)
                train_docs = set(corpus_parser.get_last_documents())
            all_docs = corpus_parser.get_documents()
    else:
        logger.info("Reloading pre-parsed dataset.")
        all_docs = Parser(session).get_documents()
        for division in ["dev", "test", "train"]:
            pdf_path = os.path.join(dirname, f"data/{division}/pdf/")
            if division == "dev":
                dev_doc_names = _files_in_dir(pdf_path)
            if division == "test":
                test_doc_names = _files_in_dir(pdf_path)
            if division == "train":
                train_doc_names = _files_in_dir(pdf_path)

        for doc in all_docs:
            if doc.name in dev_doc_names:
                dev_docs.add(doc)
            if doc.name in test_doc_names:
                test_docs.add(doc)
            if doc.name in train_doc_names:
                train_docs.add(doc)

    return all_docs, train_docs, dev_docs, test_docs
