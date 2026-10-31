# Data

`processed/salaries_2022.csv.gz` is a cleaned extract (26,382 rows, 6 columns) of the
[2022 Stack Overflow Developer Survey](https://survey.stackoverflow.co/2022/). It keeps
full-time professional developers who reported annual pay between $5,000 and $400,000 USD.

To rebuild it from the original file:

1. Download the 2022 results from https://survey.stackoverflow.co/ and unzip them into `data/raw/`.
2. Run `python src/prepare_data.py data/raw/survey_results_public.csv`.

## License

The Public 2022 Stack Overflow Developer Survey Results are made available under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/). Any rights in
individual contents of the database are licensed under the
[Database Contents License](https://opendatacommons.org/licenses/dbcl/1-0/). The processed
extract in this folder is a derived database and is shared under the same ODbL terms.
