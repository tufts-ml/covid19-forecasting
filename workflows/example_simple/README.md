Simple "Hello world" example to get started.

Usage
-----

To run a simulation, go to the main folder and just do

```console
$ conda activate aced_hmm
$ python -m aced_hmm.run_forecast --func_name python --config_path workflows/example_simple/config.json --output_dir workflows/example_output --output_file results-{{random_seed}}.csv --approximate None --random_seed 1001 --num_seeds 10
```



