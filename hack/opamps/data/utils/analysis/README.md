# Analysis Reproduction
In order to reproduce the analysis performed between our automatically generated
knowledge bases and Digikey's existing knowledge bases, we operate on a subset
of the already labeled `dev` and `test` datasets.

To create this new `analysis` dataset, we pull documents and gold labels from
`dev` and `test` that also appear in Digikey's set of datasheets and gold
labels. This is done by comparing filenames found in `dev` and `test` to
filenames found in Digikey's formatted gold labels.

## Digikey Gold Reproduction
To compare filenames from our dataset with Digikey's, we will format the raw
gold CSVs scraped from Digikey.com (see `digikeyscraper`) into a format
recognizable by our scoring and utility scripts. (The same format that our gold
labels are in, where: `(filename, manuf, part, attr, val) = line`)

### Download Raw CSVs
To get the raw CSVs from Digikey.com, you can run the following which will
install a `digikeyscraper` executable and run it to download the set of CSVs for
the opamps dataset:

```bash
$ cd ~/repos/
$ git clone https://github.com/nicholaschiang/digikey-scraper.git
$ cd digikey-scraper/
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
$ make
$ digikeyscraper -c ../hack/hack/opamps/data/src/csv/ -p ../hack/hack/opamps/data/src/pdf/ -cp 43 -f ffe00114 -sp -v
```

### Format Raw CSVs
Once you have the raw CSVs from Digikey.com, you can then run the following to
format those raw CSVs into our standard gold format (this is assuming that
you've already created a Python virtualenv and have `pip install -r
requirements.txt`):

```bash
$ cd ~/repos/hack/
$ source .venv/bin/activate
$ cd hack/opamps/data/utils/
$ python format_digikey.py
```

This will produce a file in `data/standard_digikey_gold.csv` that contains all
of the Digikey gold labels that have filenames that could be found in our gold.

## Our Gold Reproduction
In order to score Digikey's gold labels against our own ground truth labels, we
consolidate `dev_gold.csv` and `test_gold.csv` into one file:
`data/analysis/our_gold.csv`. To do this, run the following:

```bash
$ cd ~/repos/hack/hack/opamps/data/utils/analysis/
$ python make_gold_files.py
```

This will produce two files (`data/analysis/our_gold.csv` and
`data/analysis/digikey_gold.csv`) that will be referenced during gold label
comparison and scoring.
