# HACK: Automating the generation of HArdware Component Knowledge Bases

## Dependencies

We use a few applications that you'll need to install and be sure are on your
PATH.

For OS X using [homebrew](https://brew.sh):

```bash
$ brew install poppler
$ brew install postgresql
$ brew install libpng freetype pkg-config
```

On Debian-based distros:

```bash
$ sudo apt install libxml2-dev libxslt-dev python3-dev
$ sudo apt build-dep python-matplotlib
$ sudo apt install poppler-utils
$ sudo apt install postgresql
```

We require `poppler-utils` to be version 0.36.0 or greater (which is already
the case for Ubuntu 18.04). If you do not meet this requirement, you can also
[install poppler manually](https://poppler.freedesktop.org/).

For the Python dependencies, we recommend using a
[virtualenv](https://virtualenv.pypa.io/en/stable/). Once you have cloned the
repository, change directories to the root of the repository and run

```bash
$ virtualenv -p python3 .venv
```

Once the virtual environment is created, activate it by running

```bash
$ source .venv/bin/activate
```

Any Python libraries installed will now be contained within this virtual
environment. To deactivate the environment, simply run `deactivate`.

Then, install our package, Fonduer, and any other Python dependencies by running:

```bash
$ make dev
```

## Downloading the Datasets

Each component has its own dataset which must be downloaded before running. To
do so, navigate to each component's directory and run the download data script.
Note that you must navigate to the directory before running the script, since
the script will automatically unpack into the `data` directory.

For example, to download the Op-Amp dataset:

```bash
$ cd hack/opamps/
$ ./download_data.sh
```

Each dataset is already divided into a training, development, and testing set.
Manually annotated gold labels are provided in CSV form for the development and
testing sets.

## Running End-to-end Knowledge Base Construction

After installing all the requirements, and ensuring the necessary databases
are created, you can run each individual hardware component script.

Note that in our paper, we used a server with 4x14-core CPUs, 1TB of memory, and
NVIDIA GPUs. With this server, a run with the full datasets takes 10s of hours
for each component. In order to support running our experiments on consumer
hardware, we provide instructions that do not use a GPU, and scale back the
number of documents significantly.

We provide a command-line interface for each component. For more detailed
options, run `transistors -h`, `opamps -h`, or `circular_connectors -h` to see a
list of all possible options.

### Transistors

To run extraction from 500 train documents, and evaluate the resulting score on
the test set, you can run the following command. If `--max-docs` is not
specified, the entire dataset will be parsed. If you have an NVIDIA GPU with
CUDA support, you can also pass on the index of the GPU to use, e.g., `--gpu=0`.

```bash
$ createdb transistors
$ transistors --stg-temp-min --stg-temp-max --polarity --ce-v-max --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/transistors"
```

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
repository.


### Op Amps

```bash
$ createdb opamps
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

### Circular Connectors

```bash
$ createdb circular_connectors
$ circular_connectors --parse --first-time --max-docs 500 --parallel 4 --conn-string="postgresql://<user>:<pw>@<host>:<port>/circular_connectors"
```

#### Output
This executable will output 1 file.
1. A log file located in the `hack/circular_connectors/logs` directory, which
   will show runtimes and quality metrics.

### Troubleshooting

If you get an `FATAL: role "<username>" does not exist.` error, or an
`fe_sendauth no password supplied` error, you will need to make sure you have a
PostgreSQL user set up, and that you either have a PostgreSQL password, or have
configured postgres to accept connections without a password.

See [Fonduer's FAQ](https://fonduer.readthedocs.io/en/latest/user/faqs.html#)
for additional instructions.

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
   typical supply current discrepancies between Digi-Key's existing KB
   and our ground truth gold labels. This can be used later for manual
   discrepancy classification.
3. `hack/opamps/analysis/gain_analysis_discrepancies.csv`, a CSV file of
   typical gain bandwidth discrepancies between our automatically generated KB
   and our ground truth gold labels. This can be used later for manual
   discrepancy classification.
4. `hack/opamps/analysis/gain_digikey_discrepancies.csv`, a CSV file of
   typical gain bandwidth discrepancies between Digi-Key's existing KB and our
   ground truth gold labels. This can be used later for manual discrepancy
   classification.
5. `hack/transistors/analysis/ce_v_max_analysis_discrepancies.csv`, a CSV file of
   typical collector emitter voltage max discrepancies between our automatically
   generated KB and our ground truth gold labels. This can be used later for
   manual discrepancy classification.
6. `hack/transistors/analysis/ce_v_max_digikey_discrepancies.csv`, a CSV file of
   typical collector emitter voltage max discrepancies between Digi-Key's
   existing KB and our ground truth gold labels. This can be used later for
   manual discrepancy classification.
7. `hack/transistors/analysis/polarity_analysis_discrepancies.csv`, a CSV file of
   polarity discrepancies between our automatically generated KB and our ground
   truth gold labels. This can be used later for manual discrepancy
   classification.
8. `hack/transistors/analysis/polarity_digikey_discrepancies.csv`, a CSV file of
   polarity discrepancies between Digi-Key's existing KB and our ground truth
   gold labels. This can be used later for manual discrepancy classification.

We include these output files from a run on the complete dataset in this
repository.
