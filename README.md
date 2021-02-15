# Repository to explore information flows in a simple ATP synthase model
- [Project folder](/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/)

## Data
### Raw data
- [Raw data](/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/) 
Raw data is found in a different place, not synced to SFUvault because it struggles with the size and amount of files. 
The raw data is backed up to the NAS and an external hard drive through TimeMachine.

Raw data is the output of Fokker Planck simulations described by main.py, fpe.pyx, and utilities.pyx in the src folder.

A raw data file is labelled 'reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}_outfile.dat', where the {i} are placeholders for parameter values. A raw data file consists of 9 columns and N^2 rows. From left to right the columns contain: steady state probability distribution, equilibrium probability distribution, potential energy (no driving forces), drift vector components 2x, diffusion matrix components 4x. Each quantity is defined on N^2 gridpoints.

### Processed data
- [Processed data](/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/data)
A processed data file is labelled 'power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}_outfile.dat', where {i} are placeholders for parameter values. It has 7 columns and a varying number of rows. From left to right the columns contain: phase offset, power Fo/ input power, power F1/ output power, heat flow Fo, heat flow F1, power from Fo to F1, learning rate (l_1). A positive energy flow means energy flowing from the environment into the system, except for the power from Fo to F1 that is positive when energy flows from Fo to F1. A positive learning rate l_1 means F1 is learning about Fo. Each row has data for a different phase offset (if there are multiple rows).

Processed data is calculated from raw data using ATP_energy_transduction.py in the src folder. Raw data is taken in by the heat_work_info() function and it outputs the processed data. Make sure the input_file_name is set to folder containing the desired raw data and the system parameters match.

Older processed data is labelled 'flux_power_efficiency_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}_outfile.dat'. This has 6 columns and a varying number of rows. From left to right the columns contain: phase offset, flux Fo, flux F1, power Fo, power F1, efficiency. Each row has data for a different phase offset (if there are multiple rows).
It is calculated from raw data using ATP_energy_transduction.py in the src folder. Raw data is taken in by the flux_power_efficiency() function and it outputs the processed data.