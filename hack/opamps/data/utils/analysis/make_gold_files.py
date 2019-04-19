import csv
import os

from hack.transistors.data.utils.manipulate_gold import sort_gold


def get_filenames(filenames_file):
    filenames = set()
    with open(filenames_file, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for row in reader:
            filename = row[0]
            filenames.add(filename)
    return filenames


def trim_our_goldfile(goldfile, filenames, outfile, append=False):
    entities = list()
    with open(goldfile, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for row in reader:
            (filename, manuf, part, attr, val) = row

            # val = val.split(" ")[0].strip()

            # # Allow the double of a +/- value to be valid also.
            # if val.startswith("±"):
            # (value, unit) = val.split(" ")
            # val = str(2 * float(value[1:]))

            # # NOTE: For now, all we care about for the analysis is typ_gbp and
            # # typ_supply_current. We drop everything else.
            # # Normalize units to uA and kHz
            # if attr == "typ_gbp":
            # val = str(float(Quantity(val).real / 1000)).split(" ")[0].strip()
            # elif attr == "typ_supply_current":
            # val = str(float(Quantity(val).real / 1000)).split(" ")[0].strip()
            # else:
            # continue

            if filename in filenames:
                entities.append((filename, manuf, part, attr, val))

    # Now, write the new entities into outfile
    with open(outfile, "a") if append else open(outfile, "w") as inputcsv:
        writer = csv.writer(inputcsv)
        for row in entities:
            writer.writerow(row)
    return entities


def trim_digikey_goldfile(goldfile, filenames, outfile, append=False):
    entities = list()
    with open(goldfile, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        for row in reader:
            (filename, manuf, part, attr, val, source) = row

            # val = val.split(" ")[0].strip()

            # # Allow the double of a +/- value to be valid also.
            # if val.startswith("±"):
            # (value, unit) = val.split(" ")
            # val = str(2 * float(value[1:]))

            # # NOTE: For now, all we care about for the analysis is typ_gbp and
            # # typ_supply_current. We drop everything else.
            # # Normalize units to uA and kHz
            # if attr == "typ_gbp":
            # val = str(float(Quantity(val).real / 1000)).split(" ")[0].strip()
            # elif attr == "typ_supply_current":
            # val = str(float(Quantity(val).real / 1000)).split(" ")[0].strip()
            # else:
            # continue

            if filename in filenames:
                entities.append((filename, manuf, part, attr, val))

    # Now, write the new entities into outfile
    with open(outfile, "a") if append else open(outfile, "w") as inputcsv:
        writer = csv.writer(inputcsv)
        for row in entities:
            writer.writerow(row)
    return entities


if __name__ == "__main__":
    # First, get filenames that occur both in our dataset and in Digikey's gold
    dirname = os.path.dirname(__name__)
    digikey_goldfile = os.path.join(dirname, "../../standard_digikey_gold.csv")
    analysis_file = os.path.join(dirname, "../../analysis/filenames.csv")

    analysis_filenames = get_filenames(analysis_file)
    digikey_filenames = get_filenames(digikey_goldfile)
    common_filenames = digikey_filenames.intersection(analysis_filenames)

    # Then, read in our `dev` and `test` gold to create an
    # `analysis/our_gold.csv` file that contains only valid filenames.
    dev_gold = os.path.join(dirname, "../../dev/dev_gold.csv")
    test_gold = os.path.join(dirname, "../../test/test_gold.csv")
    outfile = os.path.join(dirname, "../../analysis/our_gold.csv")
    trim_our_goldfile(dev_gold, common_filenames, outfile)
    trim_our_goldfile(test_gold, common_filenames, outfile, append=True)

    # Finally, read in the formatted `standard_digikey_gold.csv` file to create
    # an `analysis/digikey_gold.csv` file that also contains only valid
    # filenames.
    digikey_outfile = os.path.join(dirname, "../../analysis/digikey_gold.csv")
    trim_digikey_goldfile(digikey_goldfile, common_filenames, digikey_outfile)

    # Sort the new gold files for comparison
    sort_gold(outfile, replace=True)
    sort_gold(digikey_outfile, replace=True)
