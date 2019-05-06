# Automating the Generation of HArdware Component Knowledge Bases (HACK)

## Minimum Software Requirements

This software artifact and the instructions provided below assume that the
system is running at minimum:

- Ubuntu 16.04.03
- Python 3.6
- PostgreSQL 9.6.9

While this software artifact can be made to work with earlier versions of Ubuntu
or Python, doing so may require additional dependencies or modifications which
are out of scope for this artifact.

## Dependencies

On Ubuntu, you'll need to ensure that the following system packages are
installed.

```bash
$ sudo apt install build-essential curl
$ sudo apt install libxml2-dev libxslt-dev python3-dev
$ sudo apt build-dep python-matplotlib
$ sudo apt install poppler-utils postgresql postgresql-contrib
$ sudo apt install virtualenv
```

We require `poppler-utils` to be version 0.36.0 or greater (which is already
the case for Ubuntu 18.04). If you do not meet this requirement, you can also
[install poppler manually](https://poppler.freedesktop.org/).

For the Python dependencies, use a
[virtualenv](https://virtualenv.pypa.io/en/stable/). This greatly simplifies
managing Python dependencies and ensures that our installation scripts work out
of the box (e.g., a python 3 `virtualenv` will automatically alias `pip` to
`pip3`). Once you have cloned the repository, change directories to the root of
the repository and run:

```bash
$ virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
$ source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install our package, [Fonduer](https://github.com/hazyresearch/fonduer),
and any other Python dependencies by running:

```bash
$ make dev
```

## Downloading the Datasets

Each component has its own dataset which must be downloaded before running. To
do so, navigate to each component's directory and run the download data script.
Note that you must navigate to the directory before running the script, since
the script will automatically unpack into the `data` directory.

For example, to download the Operation Amplifier dataset:

```bash
$ cd hack/opamps/
$ ./download_data.sh
```

Each dataset is already divided into a training, development, and testing set.
Manually annotated gold labels are provided in CSV form for the development and
testing sets.

## Setting up PostgreSQL

Make sure you have a PostgreSQL user set up. To create a user named `demo`, who
has the password `demo`, you can run the following:

```
$ psql -c "create user demo with password 'demo' superuser createdb;" -U postgres
```

## Running End-to-end Knowledge Base Construction

Note that in our paper, we used a server with 4x14-core CPUs, 1TB of memory, and
NVIDIA GPUs. With this server, a run with the full datasets takes 10s of hours
for each component. In order to support running our experiments on consumer
hardware, we provide instructions that do not use a GPU, and scale back the
number of documents significantly.

We provide a command-line interface for each component. For more detailed
options, run `transistors -h`, `opamps -h`, or `circular_connectors -h` to see a
list of all possible options.

_Note: Using SQLAlchemy with PostgreSQL has a known concurrency issue with high
levels of parallelism. Please do not use a parallel value larger than 16_.

### Transistors Dataset

To run extraction from 500 train documents, and evaluate the resulting score on
the test set, you can run the following command. If you made the demo user, you
can use `demo` as the `<user>` and `<pw>`, and the default `<host>` and `<port>`
is `localhost:5432` for PostgreSQL. The `--stg-temp-min`, `--stg-temp-max`,
`--polarity`, and `--ce-v-max` arguments represent which relations to extract
from the dataset.

```bash
$ psql -c "create database transistors with owner demo;" -U postgres
$ transistors --stg-temp-min --stg-temp-max --polarity --ce-v-max --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/transistors"
```

If `--max-docs` is not specified, the entire dataset will be parsed. If you have
an NVIDIA GPU with CUDA support, you can also pass on the index of the GPU to
use, e.g., `--gpu=0`.

#### Output
This executable will output 5 files.
1. A log file located in the `hack/transistors/logs` directory, which will show
   runtimes and quality metrics.
2. `hack/transistors/ce_v_max_dev_probs.csv`, a CSV file of maximum
   collector-emitter voltage entities from the development set and their
   corresponding probabilities, which is used later in analysis.
3. `hack/transistors/ce_v_max_test_probs.csv`, a CSV file of maximum
   collector-emitter voltage entities from the test set and their corresponding
   probabilities, which is used later in analysis.
4. `hack/transistors/polarity_dev_probs.csv`, a CSV file of polarity entities
   from the development set and their corresponding probabilities, which is used
   later in analysis.
5. `hack/transistors/polarity_test_probs.csv`, a CSV file of polarity entities
   from the test set and their corresponding probabilities, which is used
   later in analysis.

We include these output files from a run on the complete dataset in this
repository so that you can run analysis scripts using our results, without
needing to run end-to-end extraction yourself.

### Operational Amplifier Dataset

To run extraction from 500 train documents, and evaluate the resulting score on
the test set, you can run the following command. If you made the demo user, you
can use `demo` as the `<user>` and `<pw>`, and the default `<host>` and `<port>`
is `localhost:5432` for PostgreSQL.

```bash
$ psql -c "create database opamps with owner demo;" -U postgres
$ opamps --gain --current --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/opamps"
```

#### Output
This executable will output 7 files.
1. A log file located in the `hack/opamps/logs` directory, which will show
   runtimes and quality metrics.
2. `hack/opamps/current_dev_probs.csv`, a CSV file of quiescent current entities
   from the development set and their corresponding probabilities, which is used
   later in analysis.
3. `hack/opamps/current_test_probs.csv`, a CSV file of quiescent current
   entities from the test set and their corresponding probabilities, which is
   used later in analysis.
4. `hack/opamps/gain_dev_probs.csv`, a CSV file of gain bandwidth product
   entities from the development set and their corresponding probabilities,
   which is used later in analysis.
5. `hack/opamps/gain_test_probs.csv`, a CSV file of gain bandwidth product
   entities from the test set and their corresponding probabilities, which is
   used later in analysis.
6. `hack/opamps/output_current.csv`, a CSV file of quiescent current entities
   from all of the parsed documents and their corresponding probabilities, which
   is used to generate Figure 6.
7. `hack/opamps/output_gain.csv`, a CSV file of gain bandwidth product entities
   from all of the parsed documents and their corresponding probabilities, which
   is used to generate Figure 6.

We include these output files from a run on the complete dataset in this
repository.

### Circular Connectors Dataset

To run extraction from 500 train documents, and evaluate the resulting score on
the test set, you can run the following command. If you made the demo user, you
can use `demo` as the `<user>` and `<pw>`, and the default `<host>` and `<port>`
is `localhost:5432` for PostgreSQL.

```bash
$ psql -c "create database circular_connectors with owner demo;" -U postgres
$ circular_connectors --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/circular_connectors"
```

#### Output
This executable will output 1 file.
1. A log file located in the `hack/circular_connectors/logs` directory, which
   will show runtimes and quality metrics.

## Scaling Experiments
We provide two scripts in the `scripts/` directory: `scaling_docs.sh` and
`scaling_rels.sh`, which can be run to generate runtime logs used to create
runtime figures when scaling the number of documents or number of relations.
This scripts are easy to customize based on the machine being used.

## Analysis
For our analysis, we create a set of entities from our generated knowledge bases
which are then scored against ground-truth gold labels. For a more direct
comparison, we only consider a subset of datasheets which we verify are
available on Digi-Key. To evaluate on this dataset, run the following:

```bash
$ analysis --ce-v-max --polarity --gain --current
```

This will output 2 sets of scores per relation: one for our automatically
generated KB entities (shown as "Scores for cands above threshold.") and one for
entities from Digi-Key's existing KB.

### Output
This executable will output 8 files (2 per relation):
1. `hack/opamps/analysis/current_analysis_discrepancies.csv`, a CSV file of
   typical supply current discrepancies between our automatically generated KB
   and our ground truth gold labels. This can be used later for manual
   discrepancy classification.
2. `hack/opamps/analysis/current_digikey_discrepancies.csv`, a CSV file of
   typical supply current discrepancies between Digi-Key's existing KB and our
   ground truth gold labels. This can be used later for manual discrepancy
   classification.
3. `hack/opamps/analysis/gain_analysis_discrepancies.csv`, a CSV file of typical
   gain bandwidth discrepancies between our automatically generated KB and our
   ground truth gold labels. This can be used later for manual discrepancy
   classification.
4. `hack/opamps/analysis/gain_digikey_discrepancies.csv`, a CSV file of typical
   gain bandwidth discrepancies between Digi-Key's existing KB and our ground
   truth gold labels. This can be used later for manual discrepancy
   classification.
5. `hack/transistors/analysis/ce_v_max_analysis_discrepancies.csv`, a CSV file
   of typical collector emitter voltage max discrepancies between our
   automatically generated KB and our ground truth gold labels. This can be used
   later for manual discrepancy classification.
6. `hack/transistors/analysis/ce_v_max_digikey_discrepancies.csv`, a CSV file of
   typical collector emitter voltage max discrepancies between Digi-Key's
   existing KB and our ground truth gold labels. This can be used later for
   manual discrepancy classification.
7. `hack/transistors/analysis/polarity_analysis_discrepancies.csv`, a CSV file
   of polarity discrepancies between our automatically generated KB and our
   ground truth gold labels. This can be used later for manual discrepancy
   classification.
8. `hack/transistors/analysis/polarity_digikey_discrepancies.csv`, a CSV file of
   polarity discrepancies between Digi-Key's existing KB and our ground truth
   gold labels. This can be used later for manual discrepancy classification.

We include these output files from a run on the complete dataset in this
repository.

## Plotting

Finally, we include `plot_opo.py` in the `scripts/` directory, which can be used
to generate the figure of our extracted data against Digi-key's. To do so, this
script leverages the intermediate CSV files output from the `opamps` program.
Because we include intermediate files from a full run in this repository, this
plot can be generated without needing to re-run end-to-end knowledge base
construction.

## Example: End-to-End run on Transistors

As an example of how this looks end-to-end, assuming that all of the software
dependencies have been installed, these would be the commands run to go from a
fresh clone of the repository to running the results on the transistor dataset.

```
$ git clone https://github.com/lukehsiao/lctes-p27.git
$ cd lctes-p27
$ virtualenv -p python3 .venv
$ source .venv/bin/activate
(.venv) $ psql -c "create user demo with password 'demo' superuser createdb;" -U postgres
(.venv) $ psql -c "create database transistors with owner demo;" -U postgres
(.venv) $ make dev
(.venv) $ cd hack/transistors
(.venv) $ ./download_data.sh
(.venv) $ transistors --stg-temp-min --stg-temp-max --polarity --ce-v-max --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://demo:demo@localhost:5432/transistors"
```

This will produce a log output resembling:

```
[2019-05-01 08:10:17,322][WARNING] hack.transistors.transistors:212 - Parse Time (min): 7.3
[2019-05-01 08:17:45,978][WARNING] hack.transistors.transistors:334 - Candidate Extraction Time (min): 7.5
[2019-05-01 08:26:41,775][WARNING] hack.transistors.transistors:389 - Featurization Time (min): 8.7
[2019-05-01 08:56:54,200][WARNING] hack.transistors.transistors:457 - Supervision Time (min): 30.2
[2019-05-01 08:57:27,007][WARNING] hack.transistors.transistors:134 - ===================================================
[2019-05-01 08:57:27,007][WARNING] hack.transistors.transistors:135 - Entity-Level Gold Data score for stg_temp_min, b=0.808
[2019-05-01 08:57:27,007][WARNING] hack.transistors.transistors:136 - ===================================================
[2019-05-01 08:57:27,007][WARNING] hack.transistors.transistors:137 - Corpus Precision 1.000
[2019-05-01 08:57:27,007][WARNING] hack.transistors.transistors:138 - Corpus Recall    0.530
[2019-05-01 08:57:27,008][WARNING] hack.transistors.transistors:139 - Corpus F1        0.693
[2019-05-01 08:57:27,008][WARNING] hack.transistors.transistors:140 - ---------------------------------------------------
[2019-05-01 08:57:27,008][WARNING] hack.transistors.transistors:142 - TP: 53 | FP: 0 | FN: 47
[2019-05-01 08:57:27,008][WARNING] hack.transistors.transistors:146 - ===================================================
...
```

Syntax warnings/errors during the parsing phase are expected, and will not crash
the program. The `-v` verbosity flag can be included to get additional output.

## Reference

```
@inproceedings{hsiao2019hack,
  title={Automating the Generation of Hardware Component Knowledge Bases},
  author={Luke Hsiao and Sen Wu and Nicholas Chiang and Christopher Ré and Philip Levis},
  booktitle={Proceedings of the 20th ACM SIGPLAN/SIGBED Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES '19)}
  year={2019},
  month={June},
  day={23},
  organization={ACM}
}
```
