# gpu_se
Contains GPU accelerated state estimators, an MPC implementaton,
and a bioreactor model.
This code forms the practical part of my [masters thesis](https://github.com/darren-roos/thesis).

## Instillation
The following steps are a guide to getting the code running.
Version numbers shown in brackets, are the version numbers used for development.
Other versions might work as well, but have not been tested.
Where exact versions are needed, this is explicitly stated.

1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (v4.8.3)

2. Install the [CUDA 10.2 toolkit](https://developer.nvidia.com/cuda-10.2-download-archive)
    (Note: the exact version is needed)

3. Clone this repository to your computer
    ```shell script
    git clone https://github.com/darren-roos/gpu_se
    ```

4. From within the `gpu_se` directory, clone the cache repository
    ```shell script
    git clone https://github.com/darren-roos/picklejar
    ```
   
   Ensure that it is in the `gpu_se` directory:
   ```
   gpu_se
   |
   +--picklejar\
   ```

5. From within the `gpu_se` directory, create the conda environment:
   ```shell script
   conda env create -f environment.yml
   ```
   
6. Add the `gpu_se` directory to your `PYTHONPATH`

## Running

1. Activate the `conda` environment
   ```shell script
   conda activate gpu_se_cuda102
   ```
   
2. Add gpu_se to the `PYTHONPATH` environment variable.
   On linux systems the following command should work if you are in the `gpu_se`
   directory:
   ```shell script
   export PYTHONPATH=$PYTHONPATH:$pwd
   ```

3. Run any script using:
   ```shell script
   python path/to/script
   ```

The following scripts produce the results found in the thesis document:

- `results/bioreactor_closedloop/no_noise` - Shows a closedloop simulation with no noise
- `results/bioreactor_closedloop/with_noise` - Shows a closedloop simulation with noise
- `results/bioreactor_closedloop/performance_vs_control_period` - Shows the effect of control period on performance
- `results/bioreactor_closedloop/mpc_run_seq` - Benchmarks the MPC code
- `results/bioreactor_openloop/batch_production_growth` - Shows the batch, production and growth phases of the bioreactor
- `results/bioreactor_openloop/ss2ss` - Shows the open loop transition between two steady states
- 

## Building the docs
Complete documentation for the code can be built.
It requires a working instillation of `latexpdf`.
Navigate to `gpu_se/docs` and run:

```shell script
make clean latexpdf
```

The required documentation will be found in `docs/build/latex/gpuacceleratedstateestimators.pdf`.