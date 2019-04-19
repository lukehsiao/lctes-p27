import logging
import os
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Make a list of filenames to write
    filenames = set()
    for filename in os.listdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../analysis/pdf/")
    ):
        if not filename.endswith(".pdf") and not filename.endswith(".PDF"):
            logger.error(f"Invalid filename {filename}")
            pdb.set_trace()
        if filename in filenames:
            logger.error(f"Duplicate filename {filename}")
            pdb.set_trace()
        filenames.add(filename.replace(".pdf", "").replace(".PDF", ""))
        logger.debug(f"Filename {filename} is valid")

    # Write filenames to CSV
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../analysis/filenames.csv"
        ),
        "w",
    ) as outfile:
        for filename in filenames:
            outfile.write(str(filename) + "\n")
