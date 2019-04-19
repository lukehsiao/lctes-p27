import csv
import logging
import os

from hack.transistors.data.utils.analysis.make_dataset_from_list import move_files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def dissolve_dataset(filenames_file, dev_path, test_path, analysis_path):
    # Define paths
    dev_gold = os.path.join(dev_path, "dev_gold.csv")
    test_gold = os.path.join(test_path, "test_gold.csv")

    # Get a set of all filenames found in filenames_file
    filenames = get_filenames(filenames_file)
    dev_filenames = filenames.intersection(get_filenames(dev_gold))
    test_filenames = filenames.intersection(get_filenames(test_gold))

    # Move filenames from analysis/ back into corresponding dirs
    move_files(test_filenames, analysis_path, test_path)
    move_files(dev_filenames, analysis_path, dev_path)


def get_filenames(filename_file):
    filenames = set()
    with open(filename_file, "r") as inputfile:
        reader = csv.reader(inputfile)
        for line in reader:
            filenames.add(line[0])
    return filenames


if __name__ == "__main__":
    # Get filenames to move back into dev and test
    dirname = os.path.dirname(__name__)
    filenames_file = os.path.join(dirname, "../../analysis/filenames.csv")
    test_path = os.path.join(dirname, "../../test/")
    dev_path = os.path.join(dirname, "../../dev/")
    analysis_path = os.path.join(dirname, "../../analysis/")

    dissolve_dataset(filenames_file, dev_path, test_path, analysis_path)
