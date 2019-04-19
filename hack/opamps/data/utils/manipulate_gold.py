import csv
import logging
import operator
import pdb

logger = logging.getLogger(__name__)


def sort_gold(gold_file):
    # Change `output` to the absolute path of where you want the sorted output to
    # be written.
    gold_filename = gold_file.split("/")[-1]
    output = gold_file.replace(gold_filename, "") + "sorted_" + gold_filename
    data = csv.reader(open(gold_file), delimiter=",")

    # 0 specifies according to first column we want to sort (i.e. filename)
    sortedlist = sorted(data, key=operator.itemgetter(0))
    # Write sorted data into output file
    with open(output, "w") as f:
        fileWriter = csv.writer(f, delimiter=",")
        for row in sortedlist:
            fileWriter.writerow(row)

    return output


def split_gold(combinedfile, devfile, testfile, devoutfile, testoutfile):
    with open(combinedfile, "r") as combined, open(devfile, "r") as dev, open(
        testfile, "r"
    ) as test, open(devoutfile, "w") as devout, open(testoutfile, "w") as testout:
        combinedreader = csv.reader(combined)
        devreader = csv.reader(dev)
        testreader = csv.reader(test)
        devwriter = csv.writer(devout)
        testwriter = csv.writer(testout)

        # Make a set for dev filenames
        devfilenames = set()
        for line in devreader:
            filename = line[0]
            devfilenames.add(filename)

        # Make a set for test filenames
        testfilenames = set()
        for line in testreader:
            filename = line[0]
            testfilenames.add(filename)

        # Read in the combined gold and split it into dev and test files
        line_num = 0
        for line in combinedreader:
            line_num += 1
            filename = line[0]
            if filename in devfilenames:
                devwriter.writerow(line)
            elif filename in testfilenames:
                testwriter.writerow(line)
            else:
                print(f"[ERROR]: Invalid filename {filename} on line {line_num}")
                pdb.set_trace()


def combine_csv(combined, dev, test):
    with open(combined, "w") as out, open(dev, "r") as in1, open(test, "r") as in2:
        reader1 = csv.reader(in1)
        reader2 = csv.reader(in2)
        writer = csv.writer(out)
        for line in reader1:
            writer.writerow(line)
        for line in reader2:
            writer.writerow(line)
