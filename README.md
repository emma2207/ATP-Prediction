# ATP-Prediction
[Project folder](https://github.com/emma2207/ATP-Prediction)

Repository of project to investigate energy and information flows in a simple ATP synthase model.


## Src
[Code](https://github.com/emma2207/ATP-Prediction/src)

### Fokker-Planck simulations
- [main.py](https://github.com/emma2207/ATP-Prediction/src/main.py)
Main simulation file. It uses functions from fpe.pyx.
- [fpe.pyx](https://github.com/emma2207/ATP-Prediction/src/fpe.pyx) 
Actual simulation in cython to speed it up.
- [setup.py](https://github.com/emma2207/ATP-Prediction/src/setup.py) 
Compiling the code

To compile the code navigate to the folder containing these files, and in a terminal run 
'python setup.py build\_ext= --inplace'. 


#### Running and submitting jobs
- [production_slurm.sh](https://github.com/emma2207/ATP-Prediction/src/production_slurm.sh)
Necessary info to submit job to the cluster.
- [parallelize.sh](https://github.com/emma2207/ATP-Prediction/src/parallelize.sh)
Script to build the folder structure for all the jobs you want to run and copy the simulation files into each folder.
- [parallel_submit_slurm.sh](https://github.com/emma2207/ATP-Prediction/src/parallel_submit_slurm.sh)
Script to submit jobs to the cluster.

In order to get the code to compile on the cluster use 'module load gcc/5.4.0' and 'module load scipy-stack/2019a' 
before compiling using the setup file.

 
### Analysis
- [ATP\_energy\_transduction.py](https://github.com/emma2207/ATP-Prediction/src/ATP_energy_transduction.py) 
Data processing code, plotting code. Focused on energetic quantities.
- [InformationTheoreticQuantities.py](https://github.com/emma2207/ATP-Prediction/src/InformationTheoreticQuantities.py)
Plotting code, focused on information theoretic quantities.
- [utilities.pyx](https://github.com/emma2207/ATP-Prediction/src/utilities.pyx)
Supporting functions for data processing and plotting.


## Data
### Raw data
Raw data is found at '/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/'

Raw data is found in a different place, not synced to SFUvault because it struggles with the size and amount of files. 
The raw data is backed up to the NAS and an external hard drive through TimeMachine.

Raw data is the output of Fokker Planck simulations described by main.py, fpe.pyx, and utilities.pyx in the src folder.

A raw data file is labelled 
'reference\_E0\_{0}\_Ecouple\_{1}\_E1\_{2}\_psi1\_{3}\_psi2\_{4}\_n1\_{5}\_n2\_{6}\_phase\_{7}\_outfile.dat', 
where the {i} are placeholders for parameter values. A raw data file consists of 9 columns and N^2 rows. From left to 
right the columns contain: steady state probability distribution, equilibrium probability distribution, potential 
energy (no driving forces), drift vector components 2x, diffusion matrix components 4x. Each quantity is defined on N^2 
gridpoints.

### Processed data
[Processed data](https://github.com/emma2207/ATP-Prediction/data)

A processed data file is labelled 
'power\_heat\_info\_E0\_{0}\_E1\_{1}\_psi1\_{2}\_psi2\_{3}\_n1\_{4}\_n2\_{5}_\Ecouple\_{6}\_outfile.dat', 
where {i} are placeholders for parameter values. It has 7 columns and a varying number of rows. From left to right the 
columns contain: phase offset, power Fo/ input power, power F1/ output power, heat flow Fo, heat flow F1, power from Fo 
to F1, learning rate (l\_1). A positive energy flow means energy flowing from the environment into the system, except 
for the power from Fo to F1 that is positive when energy flows from Fo to F1. A positive learning rate l\_1 means F1 is 
learning about Fo. Each row has data for a different phase offset (if there are multiple rows).

Processed data is calculated from raw data using ATP\_energy\_transduction.py in the src folder. Raw data is taken in 
by the heat\_work\_info() function and it outputs the processed data. Make sure the input\_file\_name is set to folder 
containing the desired raw data and the system parameters match.

#### Old
Older processed data is labelled 
'flux\_power\_efficiency\_E0\_{0}\_E1\_{1}\_psi1\_{2}\_psi2\_{3}\_n1\_{4}\_n2\_{5}\_Ecouple\_{6}\_outfile.dat'. This 
has 6 columns and a varying number of rows. From left to right the columns contain: phase offset, flux Fo, flux F1, 
power Fo, power F1, efficiency. Each row has data for a different phase offset (if there are multiple rows).
It is calculated from raw data using ATP\_energy\_transduction.py in the src folder. Raw data is taken in by the 
flux\_power\_efficiency() function and it outputs the processed data.


## Results
[Results](https://github.com/emma2207/ATP-Prediction/results)

Mostly plots, mostly work in progress. Python plots produced by InformationTheoreticQuantities.py or 
ATP\_energy\_transduction.py from the src folder. 

## Doc
[Documents](https://github.com/emma2207/ATP-Prediction/doc)

Collection of documents and plots related to the project. To edit documents see Overleaf.
