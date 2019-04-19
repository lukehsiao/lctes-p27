import csv
import pickle

parts_by_doc = dict()

with open("dev/dev_gold.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        doc_name = row[0].upper()
        part_num = row[2]
        if doc_name not in parts_by_doc:
            parts_by_doc[doc_name] = set()
        parts_by_doc[doc_name].add(part_num)

with open("test/test_gold.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader:
        doc_name = row[0].upper()
        part_num = row[2]
        if doc_name not in parts_by_doc:
            parts_by_doc[doc_name] = set()
        parts_by_doc[doc_name].add(part_num)


print(parts_by_doc)
pickle.dump(parts_by_doc, open("parts_by_doc_new.pkl", "wb"))
