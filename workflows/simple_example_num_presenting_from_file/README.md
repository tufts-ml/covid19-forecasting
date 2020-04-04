Examples reading in multiple input files providing incoming counts:

Suppose we have these files in our current directory:
* presenting-random_seed=101.csv
* presenting-random_seed=201.csv
* presenting-random_seed=301.csv

We'd like to run a separate forecast for each one of them.

We can do that using the Snakemake file here in this workflow, with

$ snakemake --cores 1 all_wildcard_results

Which should produce in this current directory:

* results-random_seed=101.csv
* results-random_seed=201.csv
* results-random_seed=301.csv
