import csv
import logging
import os
import pdb

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset_filenames(filenames_file, testpath, devpath):
    # Generate list of filenames
    filenames = set()
    with open(filenames_file, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for row in reader:
            filename = row[0]
            filenames.add(filename)

    # Get test filenames (filenames found in testpath
    test_filenames = set()
    pdfpath = os.path.join(testpath, "pdf/")
    for filename in filenames:
        pdf = filename + ".pdf"
        if pdf not in os.listdir(pdfpath):
            logger.debug(f"Filename {pdf} not in {pdfpath}, skipping.")
        else:
            test_filenames.add(filename)

    # Get dev filenames (filenames found in devpath)
    dev_filenames = set()
    pdfpath = os.path.join(devpath, "pdf/")
    for filename in filenames:
        pdf = filename + ".pdf"
        if pdf not in os.listdir(pdfpath):
            logger.debug(f"Filename {pdf} not in {pdfpath}, skipping.")
        else:
            dev_filenames.add(filename)

    # Double check filename consistency
    if sorted(list(dev_filenames)) != sorted(
        list(filenames.difference(test_filenames))
    ):
        logger.error(f"Filenames are not consistent.")
        pdb.set_trace()

    # Return final filename sets
    return (filenames, test_filenames, dev_filenames)


def move_files(filenames, origpath, endpath):
    for filename in tqdm(filenames):
        pdf = filename + ".pdf"
        html = filename + ".html"
        os.rename(
            os.path.join(origpath, "pdf/" + pdf), os.path.join(endpath, "pdf/" + pdf)
        )
        os.rename(
            os.path.join(origpath, "html/" + html),
            os.path.join(endpath, "html/" + html),
        )


if __name__ == "__main__":

    # CSV of filenames in analysis dataset
    dirname = os.path.dirname(__name__)
    filenames_file = os.path.join(dirname, "../../analysis/filenames.csv")

    # Define dataset locations
    testpath = os.path.join(dirname, "../../test/")
    devpath = os.path.join(dirname, "../../dev/")
    endpath = os.path.join(dirname, "../../analysis/")

    # Get target filenames
    (filenames, test_filenames, dev_filenames) = get_dataset_filenames(
        filenames_file, testpath, devpath
    )

    # Move files
    move_files(test_filenames, testpath, endpath)
    move_files(dev_filenames, devpath, endpath)
