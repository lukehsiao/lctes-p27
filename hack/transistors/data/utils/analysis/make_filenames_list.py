import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_filenames_from_dir(dirname):
    filenames = set()
    for filename in os.listdir(dirname):
        if not filename.endswith(".pdf") and not filename.endswith(".PDF"):
            logger.warn(f"Invalid filename {filename}, skipping.")
        if filename in filenames:
            logger.warn(f"Duplicate filename {filename}, skipping.")
        filenames.add(filename.replace(".pdf", "").replace(".PDF", ""))
        logger.debug(f"Filename {filename} is valid")
    return filenames


if __name__ == "__main__":

    # Make a list of filenames to write
    filenames = get_filenames_from_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../analysis/pdf/")
    )

    # Write filenames to CSV
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../analysis/filenames.csv"
        ),
        "w",
    ) as outfile:
        for filename in filenames:
            outfile.write(str(filename) + "\n")
