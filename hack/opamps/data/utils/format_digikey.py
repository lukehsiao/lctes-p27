"""
This script takes in a path containing Digikey's raw gold CSVs. It then reads in
each raw CSV, formats it's gold data where:
(filename, manuf, part_num, attribute_name, value) = line
and writes the combined output to the specified output CSV.
"""

import csv
import logging
import os

from hack.opamps.data.utils.normalizers import general_normalizer, opamp_part_normalizer
from hack.opamps.data.utils.preprocessors import (
    preprocess_doc,
    preprocess_gbp,
    preprocess_manuf,
    preprocess_supply_current,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def format_digikey_gold(
    raw_gold_file, formatted_gold_file, seen, append=False, filenames="standard"
):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        for line in reader:
            """
            A line of Digikey gold data (we only care about some of it):
            """
            (
                datasheet_url,
                image_url,
                digikey_part_num,
                manuf_part_num,
                manuf,
                description,
                qty_available,
                factory_stock,
                price,
                at_qty,
                min_qty,
                packaging,
                series,
                status,
                amp_type,
                num_circuits,
                output_type,
                slew_rate,
                gain_band_prod,
                bandwidth,
                current_input_bias,
                voltage_input_offset,
                current_supply,
                current_output,
                voltage_single_dual_supply,
                op_temp,
                mount_type,
                case,
                supplier_package,
            ) = line

            """
            Extract all useful Digikey values (even if we don't have them):
            NOTE: Preprocessing returns each attribute's value, a space,
            and it's unit.
            NOTE: Normalization renders all CSVs the same (i.e. set units).
            """

            manuf = preprocess_manuf(manuf)
            part_num = opamp_part_normalizer(manuf_part_num)
            doc_name = preprocess_doc(
                manuf, part_num, datasheet_url, docformat=filenames
            )

            # For analysis purposes, skip datasheets that `get_docs()`
            # cannot get a filename for (i.e. datasheets that are not
            # in the dev or test dataset)
            if manuf == "N/A" or manuf is None or doc_name == "N/A" or doc_name is None:
                logger.warning(
                    f"Doc not found for manuf {manuf} "
                    + f"and part {part_num}, skipping."
                )
                continue

            # Relevant data (i.e. attributes that appear in our gold labels)
            typ_supply_current = preprocess_supply_current(current_supply)
            typ_gbp = preprocess_gbp(gain_band_prod)

            # Map each attribute to its corresponding normalizer
            name_attr_norm = [  # TODO: Update normalizers
                # Data that both Digikey and our labels have:
                ("typ_supply_current", typ_supply_current, general_normalizer),
                ("typ_gbp", typ_gbp, general_normalizer),
            ]

            # Output tuples of each normalized attribute
            for name, attr, normalizer in name_attr_norm:
                if "N/A" not in attr:
                    for a in attr.split(delim):
                        if len(a.strip()) > 0:
                            source_file = str(raw_gold_file.split("/")[-1])
                            output = [
                                doc_name,
                                manuf,
                                part_num,
                                name,
                                normalizer(a),
                                source_file,
                            ]
                            if tuple(output) not in seen:
                                writer.writerow(output)
                                seen.add(tuple(output))


if __name__ == "__main__":
    # Transform the transistor dataset
    filenames = "standard"

    # Change `digikey_csv_dir` to the absolute path where Digikey's raw CSVs are
    # located.
    digikey_csv_dir = "/home/nchiang/repos/hack/hack/opamps/data/src/csv/"

    # Change `formatted_gold` to the absolute path of where you want the
    # combined gold to be written.
    formatted_gold = (
        f"/home/nchiang/repos/hack/hack/opamps/data/{filenames}_digikey_gold.csv"
    )
    seen = set()

    # Run transformation
    for i, filename in enumerate(sorted(os.listdir(digikey_csv_dir))):
        if filename.endswith(".csv"):
            raw_path = os.path.join(digikey_csv_dir, filename)
            logger.info(f"[INFO]: Parsing {raw_path}")
            if i == 0:  # create file on first iteration
                format_digikey_gold(raw_path, formatted_gold, seen, filenames=filenames)
            else:  # then just append to that same file
                format_digikey_gold(
                    raw_path, formatted_gold, seen, append=True, filenames=filenames
                )
