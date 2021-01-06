Simple demonstration of the step-down functionality with just 1 patient

Usage
-----

To run a simulation, just do

$ cd workflows/example_step_down
$ snakemake --cores 1 all

To see results and verify it works,

$ cut -d, -f1-6 ../../example_output/results-random_seed\=100?.csv | column -s, -t

*Expected output for seed 1001*

```
timestep  n_Presenting  n_InGeneralWard  n_OffVentInICU  n_OnVentInICU  n_TERMINAL
0         1             0                0               0              0
1         0             1                0               0              0
2         0             0                1               0              0
3         0             1                0               0              0
4         1             0                0               0              0
5         0             0                0               0              0
6         0             0                0               0              0
7         0             0                0               0              0
8         0             0                0               0              0
9         0             0                0               0              0
10        0             0                0               0              0
11        0             0                0               0              0
12        0             0                0               0              0
13        0             0                0               0              0
14        0             0                0               0              0
```
