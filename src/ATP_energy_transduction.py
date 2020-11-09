from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz, log, argmax, sin, amax, \
    concatenate, sort, roll, amin
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as mc
from utilities import step_probability_X, calc_flux_2, calc_derivative_pxgy, step_probability_Y
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


N = 360  # N x N grid is used for Fokker-Planck simulations
dx = 2 * math.pi / N  # spacing between gridpoints
positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
timescale = 1.5 * 10**4  # conversion factor between simulation and experimental timescale

E0 = 2.0  # barrier height Fo
E1 = 2.0  # barrier height F1
psi_1 = 4.0  # chemical driving force on Fo
psi_2 = -2.0  # chemical driving force on F1
num_minima1 = 3.0  # number of barriers in Fo's landscape
num_minima2 = 3.0  # number of barriers in F1's landscape

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])  # number of energy minima/ barriers

Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])  # coupling strengths
Ecouple_array_peak = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])
Ecouple_array_double = array([11.31, 22.63, 45.25, 90.51]) # 2.83, 5.66
Ecouple_array_quad = array([1.19, 1.68, 2.38, 3.36, 4.76, 6.73, 9.51, 13.45, 19.03, 26.91, 38.05, 53.82, 76.11, 107.63])

Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))


def calc_flux_2(p_now, drift_at_pos, diffusion_at_pos, flux_array, N, dx):
    # explicit update of the corners
    # first component
    flux_array[0, 0, 0] = (
        (drift_at_pos[0, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[0, 1, 0]*p_now[1, 0]-diffusion_at_pos[0, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 1]*p_now[0, 1]-diffusion_at_pos[1, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[0, 0, N-1] = (
        (drift_at_pos[0, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[0, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[0, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, 0, 0]*p_now[0, 0]-diffusion_at_pos[1, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[0, N-1, 0] = (
        (drift_at_pos[0, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[0, 0, 0]*p_now[0, 0]-diffusion_at_pos[0, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[1, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[0, N-1, N-1] = (
        (drift_at_pos[0, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[0, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[0, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[1, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[1, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    # second component
    flux_array[1, 0, 0] = (
        (drift_at_pos[1, 0, 0]*p_now[0, 0])
        -(diffusion_at_pos[2, 1, 0]*p_now[1, 0]-diffusion_at_pos[2, N-1, 0]*p_now[N-1, 0])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 1]*p_now[0, 1]-diffusion_at_pos[3, 0, N-1]*p_now[0, N-1])/(2.0*dx)
        )
    flux_array[1, 0, N-1] = (
        (drift_at_pos[1, 0, N-1]*p_now[0, N-1])
        -(diffusion_at_pos[2, 1, N-1]*p_now[1, N-1]-diffusion_at_pos[2, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, 0, 0]*p_now[0, 0]-diffusion_at_pos[3, 0, N-2]*p_now[0, N-2])/(2.0*dx)
        )
    flux_array[1, N-1, 0] = (
        (drift_at_pos[1, N-1, 0]*p_now[N-1, 0])
        -(diffusion_at_pos[2, 0, 0]*p_now[0, 0]-diffusion_at_pos[2, N-2, 0]*p_now[N-2, 0])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 1]*p_now[N-1, 1]-diffusion_at_pos[3, N-1, N-1]*p_now[N-1, N-1])/(2.0*dx)
        )
    flux_array[1, N-1, N-1] = (
        (drift_at_pos[1, N-1, N-1]*p_now[N-1, N-1])
        -(diffusion_at_pos[2, 0, N-1]*p_now[0, N-1]-diffusion_at_pos[2, N-2, N-1]*p_now[N-2, N-1])/(2.0*dx)
        -(diffusion_at_pos[3, N-1, 0]*p_now[N-1, 0]-diffusion_at_pos[3, N-1, N-2]*p_now[N-1, N-2])/(2.0*dx)
        )

    for i in range(1, N-1):
        # explicitly update for edges not corners
        # first component
        flux_array[0, 0, i] = (
            (drift_at_pos[0, 0, i]*p_now[0, i])
            -(diffusion_at_pos[0, 1, i]*p_now[1, i]-diffusion_at_pos[0, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[1, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[1, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[0, i, 0] = (
            (drift_at_pos[0, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[0, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[0, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[1, i, 1]*p_now[i, 1]-diffusion_at_pos[1, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[0, N-1, i] = (
            (drift_at_pos[0, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[0, 0, i]*p_now[0, i]-diffusion_at_pos[0, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[1, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[1, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[0, i, N-1] = (
            (drift_at_pos[0, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[0, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[0, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[1, i, 0]*p_now[i, 0]-diffusion_at_pos[1, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # second component
        flux_array[1, 0, i] = (
            (drift_at_pos[1, 0, i]*p_now[0, i])
            -(diffusion_at_pos[2, 1, i]*p_now[1, i]-diffusion_at_pos[2, N-1, i]*p_now[N-1, i])/(2.0*dx)
            -(diffusion_at_pos[3, 0, i+1]*p_now[0, i+1]-diffusion_at_pos[3, 0, i-1]*p_now[0, i-1])/(2.0*dx)
            )
        flux_array[1, i, 0] = (
            (drift_at_pos[1, i, 0]*p_now[i, 0])
            -(diffusion_at_pos[2, i+1, 0]*p_now[i+1, 0]-diffusion_at_pos[2, i-1, 0]*p_now[i-1, 0])/(2.0*dx)
            -(diffusion_at_pos[3, i, 1]*p_now[i, 1]-diffusion_at_pos[3, i, N-1]*p_now[i, N-1])/(2.0*dx)
            )
        flux_array[1, N-1, i] = (
            (drift_at_pos[1, N-1, i]*p_now[N-1, i])
            -(diffusion_at_pos[2, 0, i]*p_now[0, i]-diffusion_at_pos[2, N-2, i]*p_now[N-2, i])/(2.0*dx)
            -(diffusion_at_pos[3, N-1, i+1]*p_now[N-1, i+1]-diffusion_at_pos[3, N-1, i-1]*p_now[N-1, i-1])/(2.0*dx)
            )
        flux_array[1, i, N-1] = (
            (drift_at_pos[1, i, N-1]*p_now[i, N-1])
            -(diffusion_at_pos[2, i+1, N-1]*p_now[i+1, N-1]-diffusion_at_pos[2, i-1, N-1]*p_now[i-1, N-1])/(2.0*dx)
            -(diffusion_at_pos[3, i, 0]*p_now[i, 0]-diffusion_at_pos[3, i, N-2]*p_now[i, N-2])/(2.0*dx)
            )

        # for points with well defined neighbours
        for j in range(1, N-1):
            # first component
            flux_array[0, i, j] = (
                (drift_at_pos[0, i, j]*p_now[i, j])
                -(diffusion_at_pos[0, i+1, j]*p_now[i+1, j]-diffusion_at_pos[0, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[1, i, j+1]*p_now[i, j+1]-diffusion_at_pos[1, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )
            # second component
            flux_array[1, i, j] = (
                (drift_at_pos[1, i, j]*p_now[i, j])
                -(diffusion_at_pos[2, i+1, j]*p_now[i+1, j]-diffusion_at_pos[2, i-1, j]*p_now[i-1, j])/(2.0*dx)
                -(diffusion_at_pos[3, i, j+1]*p_now[i, j+1]-diffusion_at_pos[3, i, j-1]*p_now[i, j-1])/(2.0*dx)
                )


def derivative_flux(flux_array, dflux_array, N, dx):
    # explicit update of the corners
    # first component
    dflux_array[0, 0, 0] = (flux_array[0, 1, 0] - flux_array[0, N - 1, 0]) / (2.0 * dx)
    dflux_array[0, 0, N - 1] = (flux_array[0, 1, N - 1] - flux_array[0, N - 1, N - 1]) / (2.0 * dx)
    dflux_array[0, N - 1, 0] = (flux_array[0, 0, 0] - flux_array[0, N - 2, 0]) / (2.0 * dx)
    dflux_array[0, N - 1, N - 1] = (flux_array[0, 0, N - 1] - flux_array[0, N - 2, N - 1]) / (2.0 * dx)

    # second component
    dflux_array[1, 0, 0] = (flux_array[1, 0, 1] - flux_array[1, 0, N - 1]) / (2.0 * dx)
    dflux_array[1, 0, N - 1] = (flux_array[1, 0, 0] - flux_array[1, 0, N - 2]) / (2.0 * dx)
    dflux_array[1, N - 1, 0] = (flux_array[1, N - 1, 1] - flux_array[1, N - 1, N - 1]) / (2.0 * dx)
    dflux_array[1, N - 1, N - 1] = (flux_array[1, N - 1, 0] - flux_array[1, N - 1, N - 2]) / (2.0 * dx)

    for i in range(1, N - 1):
        # explicitly update for edges not corners
        # first component
        dflux_array[0, 0, i] = (flux_array[0, 1, i] - flux_array[0, N - 1, i]) / (2.0 * dx)
        dflux_array[0, i, 0] = (flux_array[0, i + 1, 0] - flux_array[0, i - 1, 0]) / (2.0 * dx)
        dflux_array[0, N - 1, i] = (flux_array[0, 0, i] - flux_array[0, N - 2, i]) / (2.0 * dx)
        dflux_array[0, i, N - 1] = (flux_array[0, i + 1, N - 1] - flux_array[0, i - 1, N - 1]) / (2.0 * dx)

        # second component
        dflux_array[1, 0, i] = (flux_array[1, 0, i + 1] - flux_array[1, 0, i - 1]) / (2.0 * dx)
        dflux_array[1, i, 0] = (flux_array[1, i, 1] - flux_array[1, i, N - 1]) / (2.0 * dx)
        dflux_array[1, N - 1, i] = (flux_array[1, N - 1, i + 1] - flux_array[1, N - 1, i - 1]) / (2.0 * dx)
        dflux_array[1, i, N - 1] = (flux_array[1, i, 0] - flux_array[1, i, N - 2]) / (2.0 * dx)

        # for points with well defined neighbours
        for j in range(1, N - 1):
            # first component
            dflux_array[0, i, j] = (flux_array[0, i + 1, j] - flux_array[0, i - 1, j]) / (2.0 * dx)
            # second component
            dflux_array[1, i, j] = (flux_array[1, i, j + 1] - flux_array[1, i, j - 1]) / (2.0 * dx)


def calc_derivative(flux_array, dflux_array, N, dx, k):
    if k == 0:
        # explicit update of the corners
        dflux_array[0, 0] = (flux_array[1, 0] - flux_array[N - 1, 0]) / (2.0 * dx)
        dflux_array[0, N - 1] = (flux_array[1, N - 1] - flux_array[N - 1, N - 1]) / (2.0 * dx)
        dflux_array[N - 1, 0] = (flux_array[0, 0] - flux_array[N - 2, 0]) / (2.0 * dx)
        dflux_array[N - 1, N - 1] = (flux_array[0, N - 1] - flux_array[N - 2, N - 1]) / (2.0 * dx)

        for i in range(1, N - 1):
            # explicitly update for edges not corners
            dflux_array[0, i] = (flux_array[1, i] - flux_array[N - 1, i]) / (2.0 * dx)
            dflux_array[i, 0] = (flux_array[i + 1, 0] - flux_array[i - 1, 0]) / (2.0 * dx)
            dflux_array[N - 1, i] = (flux_array[0, i] - flux_array[N - 2, i]) / (2.0 * dx)
            dflux_array[i, N - 1] = (flux_array[i + 1, N - 1] - flux_array[i - 1, N - 1]) / (2.0 * dx)

            # for points with well defined neighbours
            for j in range(1, N - 1):
                dflux_array[i, j] = (flux_array[i + 1, j] - flux_array[i - 1, j]) / (2.0 * dx)

    if k == 1:
        dflux_array[0, 0] = (flux_array[0, 1] - flux_array[0, N - 1]) / (2.0 * dx)
        dflux_array[0, N - 1] = (flux_array[0, 0] - flux_array[0, N - 2]) / (2.0 * dx)
        dflux_array[N - 1, 0] = (flux_array[N - 1, 1] - flux_array[N - 1, N - 1]) / (2.0 * dx)
        dflux_array[N - 1, N - 1] = (flux_array[N - 1, 0] - flux_array[N - 1, N - 2]) / (2.0 * dx)

        for i in range(1, N - 1):
            # explicitly update for edges not corners
            dflux_array[0, i] = (flux_array[0, i + 1] - flux_array[0, i - 1]) / (2.0 * dx)
            dflux_array[i, 0] = (flux_array[i, 1] - flux_array[i, N - 1]) / (2.0 * dx)
            dflux_array[N - 1, i] = (flux_array[N - 1, i + 1] - flux_array[N - 1, i - 1]) / (2.0 * dx)
            dflux_array[i, N - 1] = (flux_array[i, 0] - flux_array[i, N - 2]) / (2.0 * dx)

            # for points with well defined neighbours
            for j in range(1, N - 1):
                # second component
                dflux_array[i, j] = (flux_array[i, j + 1] - flux_array[i, j - 1]) / (2.0 * dx)


def flux_power_efficiency(target_dir):  # processing of raw data
    Ecouple_array = array([0.0, 1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    phase_array = array([0.0])

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            integrate_flux_X = empty(phase_array.size)
            integrate_flux_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            efficiency_ratio = empty(phase_array.size)

            for Ecouple in Ecouple_array:
                for ii, phase_shift in enumerate(phase_array):
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200914_reruns_n/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

                    output_file_name = (target_dir + "200914_reruns_n/" + "flux_power_efficiency_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    print("Calculating flux for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " +
                          f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                    num_minima2, phase_shift),
                                             usecols=(0, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))  # check grid size
                        dx = 2 * math.pi / N  # spacing between gridpoints
                        positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
                        print('Grid size: ', N)

                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
                        flux_array = asarray(flux_array)/(dx*dx)

                        # Note that the factor of 2 pi actually needs to be removed to get the right units.
                        # Currently, all the powers being plotted in this script are multiplied by 2 pi
                        # to make up for this factor
                        integrate_flux_X[ii] = (1/(2*pi))*trapz(trapz(flux_array[0, ...], dx=dx, axis=1), dx=dx)
                        integrate_flux_Y[ii] = (1/(2*pi))*trapz(trapz(flux_array[1, ...], dx=dx, axis=0), dx=dx)

                        integrate_power_X[ii] = integrate_flux_X[ii]*psi_1
                        integrate_power_Y[ii] = integrate_flux_Y[ii]*psi_2
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))
                if abs(psi_1) <= abs(psi_2):
                    efficiency_ratio = -(integrate_power_X/integrate_power_Y)
                else:
                    efficiency_ratio = -(integrate_power_Y/integrate_power_X)

                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as \
                        ofile:
                    for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_flux_X[ii]:.15e}" + "\t"
                            + f"{integrate_flux_Y[ii]:.15e}" + "\t"
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{efficiency_ratio[ii]:.15e}" + "\n")
                    ofile.flush()


def heat_work_info(target_dir):
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    psi1_array = array([2.0])
    psi2_array = array([-1.78, -1.33, -1.6])
    phase_array = array([0.0])

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            integrate_heat_X = empty(phase_array.size)
            integrate_heat_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            energy_o_1 = empty(phase_array.size)
            learning_rate = empty(phase_array.size)

            for Ecouple in Ecouple_array_tot:
                for ii, phase_shift in enumerate(phase_array):
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200220_moregrid/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

                    output_file_name = (target_dir + "data/200915_energyflows/" + "power_heat_info_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    print("Calculating stuff for " + f"psi_1 = {psi_1}, psi_2 = {psi_2}, " +
                          f"Ecouple = {Ecouple}, num_minima1 = {num_minima1}, num_minima2 = {num_minima2}")

                    try:
                        data_array = loadtxt(
                            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                   phase_shift)
                        )
                        N = int(sqrt(len(data_array)))  # check grid size
                        dx = 2 * math.pi / N  # spacing between gridpoints
                        positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
                        print('Grid size: ', N)

                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        prob_eq_array = data_array[:, 1].reshape((N, N))
                        potential_at_pos = data_array[:, 2].reshape((N, N))
                        drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))

                    # calculate power
                    flux_array = zeros((2, N, N))
                    calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

                    integrate_power_X[ii] = trapz(trapz(flux_array[0, ...])) * timescale * psi_1
                    integrate_power_Y[ii] = trapz(trapz(flux_array[1, ...])) * timescale * psi_2

                    # calculate heat flow
                    dpotential_x = zeros((N, N))
                    dpotential_y = zeros((N, N))
                    calc_derivative(potential_at_pos, dpotential_x, N, dx, 0)
                    calc_derivative(potential_at_pos, dpotential_y, N, dx, 1)

                    integrate_heat_X[ii] = trapz(trapz(flux_array[0, ...] * dpotential_x)) * timescale \
                                           - integrate_power_X[ii]
                    integrate_heat_Y[ii] = trapz(trapz(flux_array[1, ...] * dpotential_y)) * timescale \
                                           - integrate_power_Y[ii]

                    # calculate energy flow between subsystems
                    force_FoF1 = zeros((N, N))
                    for i in range(N):
                        for j in range(N):
                            force_FoF1[i, j] = -0.5 * Ecouple * sin(positions[i] - positions[j])

                    energy_o_1[ii] = trapz(trapz(flux_array[0, ...] * force_FoF1)) * timescale

                    # calculate learning rate
                    dflux_array = empty((2, N, N))
                    derivative_flux(flux_array, dflux_array, N, dx)

                    for i in range(N):
                        for j in range(N):
                            if prob_ss_array[i, j] == 0:
                                prob_ss_array[i, j] = 10e-18

                    learning_rate[ii] = -trapz(trapz(
                        dflux_array[1, ...] * log(prob_ss_array)
                    )) * timescale

                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as \
                        ofile:
                    for ii, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_power_X[ii]:.15e}" + "\t"
                            + f"{integrate_power_Y[ii]:.15e}" + "\t"
                            + f"{integrate_heat_X[ii]:.15e}" + "\t"
                            + f"{integrate_heat_Y[ii]:.15e}" + "\t"
                            + f"{energy_o_1[ii]:.15e}" + "\t"
                            + f"{learning_rate[ii]:.15e}" + "\n")
                    ofile.flush()


def plot_power_Ecouple(target_dir):  # plot power and efficiency vs coupling strength
    Ecouple_array_tot = array(
            [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
             45.25, 64.0, 90.51, 128.0])
    # Ecouple_array_tot = array([2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

    output_file_name = (
        "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
        "Output_power2_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_.pdf")
    f, axarr = plt.subplots(1, 1, figsize=(8, 6))
    # axarr.axhline(0, color='black')

    # zero-barrier results
    # input_file_name = (target_dir + "plotting_data/"
    #                    + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    # data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    # Ecouple_array2 = array(data_array[:, 0])
    # flux_x_array = array(data_array[:, 1])
    # flux_y_array = array(data_array[:, 2])
    # power_y = -flux_y_array * psi_2
    # power_x = flux_x_array * psi_1
    # # axarr.plot(Ecouple_array2, 2 * pi * timescale * (power_x - power_y), '-', color='C0', label='$0$', linewidth=2)
    # axarr.plot(Ecouple_array2, 2*pi*timescale * power_y, '-', color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    i = 0  # only use phase=0 data
    power_x_array = []
    power_y_array = []
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Driving_forces/" +
                           "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                           "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 3, 4))
            if Ecouple in [-1.0]:  # data format varies a little
                power_x = array(data_array[i, 1])
                power_y = array(data_array[i, 2])
            else:
                power_x = array(data_array[1])
                power_y = array(data_array[2])
            power_x_array = append(power_x_array, power_x)
            power_y_array = append(power_y_array, power_y)
        except OSError:
            print('Missing file flux')

    # axarr.plot(Ecouple_array_tot, 2*pi*timescale*(power_x_array + power_y_array), 'o', color='C1', label='$2$',
    #            markersize=8)
    axarr.plot(Ecouple_array, -2 * pi * timescale * power_y_array, 'o', color='C0', markersize=8, label='original')

    power_x_array = []
    power_y_array = []
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (target_dir + "200810_bipartite/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 3, 4))
            if Ecouple in [-1.0]:  # data format varies a little
                power_x = array(data_array[i, 1])
                power_y = array(data_array[i, 2])
            else:
                power_x = array(data_array[1])
                power_y = array(data_array[2])
            power_x_array = append(power_x_array, power_x)
            power_y_array = append(power_y_array, power_y)
        except OSError:
            print('Missing file flux')

    # axarr.plot(Ecouple_array_tot, 2*pi*timescale*(power_x_array + power_y_array), 'o', color='C1', label='$2$',
    #            markersize=8)
    axarr.plot(Ecouple_array, -2 * pi * timescale * power_y_array, 'o', color='C1', markersize=6, label='bipartite')

    axarr.yaxis.offsetText.set_fontsize(14)
    axarr.tick_params(axis='both', labelsize=16)
    axarr.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr.set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    # axarr.spines['bottom'].set_visible(False)
    axarr.set_xlim((1.7, 135))
    axarr.set_xscale('log')
    # axarr.set_ylim((-60, 31))
    # axarr.set_yticks([-50, -25, 0, 25])

    leg = axarr.legend(fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_power_efficiency_Ecouple(target_dir):  # plot power and efficiency vs coupling strength
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
         45.25, 64.0, 90.51, 128.0])
    Ecouple_array3 = array([2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.62, 32.0, 45.25, 64.0, 90.51, 128.0])

    barrier_heights = array([2.0, 4.0])
    barrier_label = ['$2$', '$4$', '$6$']
    colorlst = ['C1', 'C9']
    offset = [0, 4]

    output_file_name = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "P_ATP_eff_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 8))

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis
    maxpower = 2 * pi * 0.000085247 * timescale
    axarr[0].axhline(maxpower, color='C1', linestyle=':', linewidth=2)  # line at infinite power coupling
    axarr[0].fill_between([1, 250], 0, 31, facecolor='grey', alpha=0.2)  # shading power output

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x axis
    axarr[1].axhline(1, color='C1', linestyle=':', linewidth=2)  # max efficiency
    axarr[1].fill_between([1, 250], 0, 1, facecolor='grey', alpha=0.2)  # shading power output

    # zero-barrier results
    input_file_name = (target_dir + "plotting_data/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    Ecouple_array2 = array(data_array[:, 0])
    flux_x_array = array(data_array[:, 1])
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', label='$0$', linewidth=2)
    axarr[1].plot(Ecouple_array2, flux_y_array / flux_x_array, '-', color='C0', linewidth=2)

    # peak position estimate output power from theory
    # Ecouple_est = 3.31 + 4 * pi * (psi_1 - psi_2) / 9
    # axarr[0].axvline(Ecouple_est, color='black', linestyle='-', linewidth=2)

    # Fokker-Planck results
    i = 0  # only use phase=0 data
    for j, E0 in enumerate(barrier_heights):
        E1 = E0
        power_y_array = []
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if E0 == 2.0 and Ecouple not in Ecouple_extra:
                input_file_name = (target_dir + "plotting_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            elif E0 == 2.0 and Ecouple in Ecouple_extra:
                input_file_name = (target_dir + "200511_2kT_extra/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            elif E0 == 4.0:
                input_file_name = (target_dir + "200506_4kTbarrier/spectral/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            elif E0 == 6.0:
                input_file_name = (target_dir + "200518_6kTbarrier/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(0, 4, 5))
                if Ecouple in Ecouple_array and E0 == 2.0:  # data format varies a little
                    power_y_array = append(power_y_array, data_array[i, 1])
                    eff_array = append(eff_array, data_array[i, 2])
                else:
                    power_y_array = append(power_y_array, data_array[1])
                    eff_array = append(eff_array, data_array[2])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(4.0, 4.0, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
        axarr[0].axvline(Ecouple_array_tot[argmax(-power_y_array)], linestyle=(offset[j], (4, 4)), color=colorlst[j],
                         linewidth=2)
        axarr[1].axvline(Ecouple_array_tot[argmax(-power_y_array)], linestyle=(offset[j], (4, 4)), color=colorlst[j],
                         linewidth=2)
        axarr[0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, 'o', color=colorlst[j], label=barrier_label[j],
                      markersize=8)
        axarr[1].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), 'o', color=colorlst[j], markersize=8)

    # rate calculations theory line efficiency
    # pos = linspace(1, 128, 200)
    # theory = 1 - 3 * exp((pi / 3) * (psi_1 - psi_2) - 0.75 * pos)
    # axarr[1].plot(pos, theory, '--', color='black', linewidth=2)

    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].spines['bottom'].set_visible(False)
    axarr[0].set_xlim((1.7, 135))
    axarr[0].set_ylim((None, 31))
    # axarr[0].set_yticks([-50, -25, 0, 25])
    # axarr[0].set_yscale('log')

    leg = axarr[0].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    axarr[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_xscale('log')
    axarr[1].set_xlim((1.7, 135))
    axarr[1].set_ylim((-0.5, 1.05))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].spines['bottom'].set_visible(False)
    axarr[1].set_yticks([-0.5, 0, 0.5, 1.0])
    axarr[1].tick_params(axis='both', labelsize=14)

    # f.text(0.05, 0.95, r'$\mathbf{a)}$', ha='center', fontsize=20)
    # f.text(0.05, 0.48, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_power_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    Ecouple_array_tot = array([5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])

    output_file_name = (target_dir + "P_ATP_Ecouple_grid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='all', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # line at highest Ecouple power
            input_file_name = (
                            target_dir + "plotting_data/"
                            + "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_n1_3.0_n2_3.0_outfile.dat")
            try:
                inf_array = loadtxt(input_file_name, usecols=2)
            except OSError:
                print('Missing file Infinite Power Coupling')

            axarr[i, j].axhline(2*pi*inf_array[i*3 + j] * timescale, color='grey', linestyle=':', linewidth=1)

            # zero-barrier result
            input_file_name = (
                        target_dir + "plotting_data/"
                        + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', linewidth=3)

            # Fokker-Planck results (barriers)
            power_y_array = []
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (
                            target_dir + "plotting_data/" + "flux_power_efficiency_"
                            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=4)

                    if data_array.size > 2:  # data format varies a little
                        power_y = array(data_array[0])
                    else:
                        power_y = array(data_array)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file flux')
            axarr[i, j].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, '.', color='C1', markersize=14)

            # print('Ratio max power / infinite coupling power', amax(-power_y_array)/inf_array[i*3 + j], '\n')

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((1.6, 150))
            elif j == 1:
                axarr[i, j].set_xlim((2.3, 150))
            else:
                axarr[i, j].set_xlim((5, 150))

            if i == 0:
                axarr[i, j].set_ylim((0, 7.8))
                axarr[i, j].set_yticks([0, 3.0, 6.0])
                axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            elif i == 1:
                axarr[i, j].set_ylim((0, 31))
                axarr[i, j].set_yticks([0, 15, 30])
                axarr[i, j].set_yticklabels([r'$0$', r'$15$', r'$30$'])
            else:
                axarr[i, j].set_ylim((0, 122))
                axarr[i, j].set_yticks([0, 50, 100])
                axarr[i, j].set_yticklabels([r'$0$', r'$50$', r'$100$'])

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=18)

            if j == 2 and i == 1:
                axarr[i, j].tick_params(axis='x', colors='red', which='both')
                axarr[i, j].tick_params(axis='y', colors='red', which='both')
                axarr[i, j].spines['left'].set_color('red')
                axarr[i, j].spines['bottom'].set_color('red')
            else:
                axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', va='center', rotation='vertical',
           fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_power_efficiency_phi(target_dir): # plot power and efficiency as a function of the coupling strength
    output_file_name = (
                target_dir + "power_efficiency_phi_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 6))

    # flux plot
    axarr[0].axhline(0, color='black', linewidth=1)  # line at zero

    # zero-barrier results
    input_file_name = (
                target_dir + "plotting_data/"
                + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate([16.0]):
        input_file_name = (
                    target_dir + "plotting_data/" + "flux_power_efficiency_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = array(data_array[:, 0])
            power_y = array(data_array[:, 1])
        except OSError:
            print('Missing file flux')
    axarr[0].plot(phase_array, -2*pi*power_y*timescale, 'o', color='C1', label='$2$', markersize=8)

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='both', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, 31))
    axarr[0].set_xlim((0, 2.1))
    axarr[0].set_yticks([0, 10, 20, 30])

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(0.5)

    for ii, Ecouple in enumerate([16.0]):
        input_file_name = (
                    target_dir + "plotting_data/flux_power_efficiency_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=5)
            eff_array = data_array
        except OSError:
            print('Missing file efficiency')
    axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), 'o', color='C1', label='$2$', markersize=8)

    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_ylim((0, 1.1))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].yaxis.offsetText.set_fontsize(14)
    axarr[1].tick_params(axis='both', labelsize=14)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    leg = axarr[1].legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=14, loc='lower right', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.55, 0.07, r'$n \phi\ (\rm rad)$', fontsize=20, ha='center')
    f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
    f.text(0.03, 0.37, r'$\mathbf{b)}$', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_power_phi_single(target_dir):  # plot of the power as a function of the phase offset
    colorlst = ['C2', 'C3', 'C1', 'C4']
    markerlst = ['D', 's', 'o', 'v']
    Ecouple_array = array([2.0, 8.0, 16.0, 32.0])

    output_file_name = (target_dir
                        + "Power_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.axhline(0, color='black', linewidth=1)

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (target_dir + "plotting_data/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = data_array[:, 0]
            power_y_array = data_array[:, 1]

            ax.plot(phase_array, -2*pi*power_y_array*timescale, linestyle='-', marker=markerlst[ii],
                    label=f'${Ecouple}$', markersize=8, linewidth=2, color=colorlst[ii])
        except OSError:
            print('Missing file')

    # Infinite coupling result
    input_file_name = (target_dir + "plotting_data/"
                       + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
    data_array = loadtxt(input_file_name, usecols=(0, 1))
    phase_array = data_array[:, 0]
    power_y_array = -psi_2 * data_array[:, 1]

    ax.plot(phase_array[:61], 2*pi*power_y_array[:61]*timescale, '-', label=f'$\infty$', linewidth=2, color='C6')
    ax.tick_params(axis='both', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.set_xlim((0, 2.1))
    ax.set_ylim((-41, 31))
    ax.set_yticks([-40, -20, 0, 20])

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], title=r'$\beta E_{\rm couple}$', fontsize=14, loc=[0.8, 0.08],
                    frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.55, 0.02, r'$n \phi\ (\rm rad)$', fontsize=20, ha='center')
    plt.ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    plt.xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3],
               ['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    f.tight_layout()
    f.subplots_adjust(bottom=0.14)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_nn_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array([4.0, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    output_file_name = (
            target_dir + "power_nn_efficiency_Ecouple_plot_more_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_log_.pdf")

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=(0, 4))
                if len(data_array) == 2:  # data format varies a little
                    power_y = array(data_array[1])
                else:
                    power_y = array(data_array[0, 1])
                power_y_array = append(power_y_array, power_y)
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

        # Infinite coupling data
        input_file_name = (target_dir + "plotting_data/" +
        "Power_ATP_Ecouple_inf_no_n1_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].tick_params(axis='y', labelsize=14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0, 0].spines['right'].set_visible(False)
    axarr[0, 0].spines['top'].set_visible(False)
    axarr[0, 0].set_ylim((0, 20))
    axarr[0, 0].set_xlim((7, None))
    axarr[0, 0].set_yticks([0, 5, 10, 15, 20])

    axarr[0, 1].spines['right'].set_visible(False)
    axarr[0, 1].spines['top'].set_visible(False)
    axarr[0, 1].spines['left'].set_visible(False)
    axarr[0, 1].set_xticks([300])
    axarr[0, 1].set_xticklabels(['$\infty$'])
    axarr[0, 1].tick_params(axis='y', color='white')

    # broken axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
    axarr[0, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-d, +d), **kwargs)
    kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
    axarr[0, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-d, +d), **kwargs)

    #########################################################
    # efficiency plot
    # axarr[1, 0].axhline(0, color='black', linewidth=1, label='_nolegend_')
    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=5)
                if data_array.size == 1:  # data format varies a little
                    eff_array = append(eff_array, data_array)
                else:
                    eff_array = append(eff_array, data_array[0])
            except OSError:
                print('Missing file efficiency')

        # infinite coupling value
        axarr[1, 0].plot(Ecouple_array_tot, eff_array/0.5, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])
        axarr[1, 1].plot([300], 1, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])

    axarr[1, 0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1, 0].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1, 0].set_xscale('log')
    axarr[1, 0].set_xlim((7, None))
    axarr[1, 0].set_ylim((0, None))
    axarr[1, 0].spines['right'].set_visible(False)
    axarr[1, 0].spines['top'].set_visible(False)
    axarr[1, 0].set_yticks([0, 0.5, 1.0])
    axarr[1, 0].tick_params(axis='both', labelsize=14)
    axarr[1, 0].set_xticks([10, 100])
    axarr[1, 0].set_xticklabels(['$10^1$', '$10^2$'])

    axarr[1, 1].spines['right'].set_visible(False)
    axarr[1, 1].spines['top'].set_visible(False)
    axarr[1, 1].spines['left'].set_visible(False)
    axarr[1, 1].set_xticks([300])
    axarr[1, 1].set_xticklabels(['$\infty$'])
    axarr[1, 1].set_xlim((295, 305))
    axarr[1, 1].tick_params(axis='y', color='white')
    axarr[1, 1].tick_params(axis='x', labelsize=14)

    # broken axis
    kwargs = dict(transform=axarr[1, 0].transAxes, color='k', clip_on=False)
    axarr[1, 0].plot((1 - 0.3 * d, 1 + 0.3 * d), (-2 * d, +2 * d), **kwargs)
    kwargs.update(transform=axarr[1, 1].transAxes)  # switch to the bottom axes
    axarr[1, 1].plot((-2.5 * d - 0.05, +2.5 * d - 0.05), (-2 * d, +2 * d), **kwargs)

    leg = axarr[1, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o} = n_1$', fontsize=14,
                             loc='lower right', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.05, 0.92, r'$\mathbf{a)}$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_nn_power_efficiency_phi(target_dir):  # plot power and efficiency as a function of the coupling strength
    phase_array = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    Ecouple_array = array([16.0])
    n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$']
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 4.5))

    output_file_name = (
            target_dir + "power_efficiency_phi_vary_n_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis

    # zero-barrier results
    input_file_name = (
                target_dir + "plotting_data/"
                + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=4)
                if num_min == 3.0:
                    power_y = array(data_array[::2])
                else:
                    power_y = array(data_array)
            except OSError:
                print('Missing file flux')
        if num_min != 3.0:
            power_y = append(power_y, power_y[0])
        axarr[0].plot(phase_array, -2*pi*power_y*timescale, '-', markersize=8, color=color_lst[i],
                      marker=markerlst[i], label=n_labels[i])

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='both', labelsize=14)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, None))
    axarr[0].set_xlim((0, 6.3))

    leg = axarr[0].legend(title=r'$n_{\rm o} = n_1$', fontsize=14, loc='lower center', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    #########################################################
    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(1.5)

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=5)
                if num_min == 3.0:
                    eff_array = array(data_array[::2])
                else:
                    eff_array = array(data_array)
            except OSError:
                print('Missing file efficiency')
        if num_min != 3.0:
            eff_array = append(eff_array, eff_array[0])
        axarr[1].plot(phase_array, eff_array / (-psi_2 / psi_1), marker=markerlst[i], label=n_labels[i],
                      markersize=8, color=color_lst[i])

    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1].set_ylim((0, 1.1))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].yaxis.offsetText.set_fontsize(14)
    axarr[1].tick_params(axis='both', labelsize=14)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    f.text(0.55, 0.01, r'$n \phi \ (\rm rad)$', fontsize=20, ha='center')  # xlabel seems broken
    f.text(0.03, 0.93, r'$\mathbf{a)}$', fontsize=20)
    f.text(0.03, 0.4, r'$\mathbf{b)}$', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_n0_power_efficiency_Ecouple(target_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = array(
        [8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    output_file_name = (
                target_dir + "/power_efficiency_Ecouple_plot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}" + "_log_.pdf")
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    # power plot
    # Fokker-Planck results (barriers
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple),
                    usecols=4)
                power_y = array(data_array)
                if power_y.size == 1:  # data format varies a little
                    power_y_array = append(power_y_array, power_y)
                else:
                    power_y_array = append(power_y_array, power_y[0])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))

        axarr[0, 0].plot(Ecouple_array_tot, -2*pi*power_y_array*timescale, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])

        # infinite coupling result
        input_file_name = (target_dir + "plotting_data/" +
                           "Power_ATP_Ecouple_inf_no_varies_n1_3.0_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat"
                           )
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1}) $', fontsize=20)
    axarr[0, 0].spines['right'].set_visible(False)
    axarr[0, 0].spines['top'].set_visible(False)
    axarr[0, 0].set_ylim((0, None))
    axarr[0, 0].set_xlim((7.5, None))
    axarr[0, 0].tick_params(axis='both', labelsize=14)
    axarr[0, 0].set_yticks([0, 5, 10, 15, 20])

    axarr[0, 1].spines['right'].set_visible(False)
    axarr[0, 1].spines['top'].set_visible(False)
    axarr[0, 1].spines['left'].set_visible(False)
    axarr[0, 1].set_xticks([300])
    axarr[0, 1].set_xticklabels(['$\infty$'])
    axarr[0, 1].tick_params(axis='y', color='white')

    # broken axes
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=axarr[0, 0].transAxes, color='k', clip_on=False)
    axarr[0, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-d, +d), **kwargs)
    kwargs.update(transform=axarr[0, 1].transAxes)  # switch to the bottom axes
    axarr[0, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-d, +d), **kwargs)

    #########################################################
    # efficiency plot
    axarr[1, 0].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')  # max efficiency
    axarr[1, 1].axhline(1, color='grey', linestyle=':', linewidth=1, label='_nolegend_')

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        eff_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "plotting_data/" + "flux_power_efficiency_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple), usecols=5)
                if data_array.size == 1:
                    eff_array = append(eff_array, data_array)
                else:
                    eff_array = append(eff_array, data_array[0])
            except OSError:
                print('Missing file efficiency')

        axarr[1, 0].plot(Ecouple_array_tot, eff_array / (-psi_2 / psi_1), marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[1, 1].plot([300], [1], marker=markerlst[j], markersize=6, color=color_lst[j])  # infinite coupling

    axarr[1, 0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr[1, 0].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=20)
    axarr[1, 0].set_xscale('log')
    axarr[1, 0].set_ylim((0, None))
    axarr[1, 0].spines['right'].set_visible(False)
    axarr[1, 0].spines['top'].set_visible(False)
    axarr[1, 0].set_yticks([0, 0.5, 1.0])
    axarr[1, 0].tick_params(axis='both', labelsize=14)
    axarr[1, 0].set_xticks([10, 100])
    axarr[1, 0].set_xticklabels(['$10^1$', '$10^2$'])

    axarr[1, 1].spines['right'].set_visible(False)
    axarr[1, 1].spines['top'].set_visible(False)
    axarr[1, 1].spines['left'].set_visible(False)
    axarr[1, 1].set_xticks([300])
    axarr[1, 1].set_xticklabels(['$\infty$'])
    axarr[1, 1].set_xlim((295, 305))
    axarr[1, 1].tick_params(axis='y', color='white')
    axarr[1, 1].tick_params(axis='x', labelsize=14)

    # broken axes
    kwargs = dict(transform=axarr[1, 0].transAxes, color='k', clip_on=False)
    axarr[1, 0].plot((1 - 0.3*d, 1 + 0.3*d), (-2*d, +2*d), **kwargs)
    kwargs.update(transform=axarr[1, 1].transAxes)  # switch to the bottom axes
    axarr[1, 1].plot((-2.5*d-0.05, +2.5*d-0.05), (-2*d, +2*d), **kwargs)

    leg = axarr[1, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14,
                             loc='lower center', frameon=False, ncol=3)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.67, 0.25, r'$n_1=3$', ha='center', fontsize=14)
    f.text(0.05, 0.92, r'$\mathbf{a)}$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\mathbf{b)}$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 3.0))


def calc_heat_flow():
    phase_array = array([0.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    dt = 5e-2

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            integrate_flux_X = empty(phase_array.size)
            integrate_flux_Y = empty(phase_array.size)
            integrate_power_X = empty(phase_array.size)
            integrate_power_Y = empty(phase_array.size)
            integrate_heat_X = empty(phase_array.size)
            integrate_heat_Y = empty(phase_array.size)
            integrate_energy_X = empty(phase_array.size)
            integrate_energy_Y = empty(phase_array.size)
            integrate_couple_X = empty(phase_array.size)
            integrate_couple_Y = empty(phase_array.size)
            integrate_entropy_X = empty(phase_array.size)
            integrate_entropy_Y = empty(phase_array.size)
            integrate_entropy = empty(phase_array.size)
            integrate_shannon = empty(phase_array.size)
            for Ecouple in Ecouple_array_total:
                for ii, phase_shift in enumerate(phase_array):
                    if Ecouple in Ecouple_array:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    elif Ecouple in Ecouple_array_peak:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_Extra_measurements_Ecouple/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    elif Ecouple in Ecouple_array_double:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")

                    output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/" + "results/" +
                                        "comp_energy_flow_" +
                                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

                    try:
                        data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                    num_minima2, phase_shift),
                                             usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                        N = int(sqrt(len(data_array)))  # check grid size
                        dx = 2 * math.pi / N
                        # print('Grid size: ', N)

                        prob_ss_array = data_array[:, 0].reshape((N, N))
                        potential_at_pos = data_array[:, 1].reshape((N, N))
                        drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                        diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

                        flux_array = zeros((2, N, N))
                        # dflux_array = zeros((2, N, N))
                        calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
                        flux_array = asarray(flux_array)
                        # derivative_flux(flux_array, dflux_array, N)
                        # dflux_array = asarray(dflux_array)
                        dpotential_x = zeros((N, N))
                        dpotential_y = zeros((N, N))
                        calc_derivative(potential_at_pos, dpotential_x, N, dx, 0)
                        calc_derivative(potential_at_pos, dpotential_y, N, dx, 1)

                        integrate_flux_X[ii] = trapz(trapz(flux_array[0, ...], dx=dx), dx=dx) * timescale
                        integrate_flux_Y[ii] = trapz(trapz(flux_array[1, ...], dx=dx), dx=dx) * timescale

                        integrate_power_X[ii] = psi_1*integrate_flux_X[ii]
                        integrate_power_Y[ii] = psi_2*integrate_flux_Y[ii]

                        integrate_energy_X[ii] = trapz(trapz(flux_array[0, ...] * dpotential_x, dx=dx), dx=dx) * timescale
                        integrate_energy_Y[ii] = trapz(trapz(flux_array[1, ...] * dpotential_y, dx=dx), dx=dx) * timescale

                        integrate_heat_X[ii] = - integrate_power_X[ii] + integrate_energy_X[ii]
                        integrate_heat_Y[ii] = - integrate_power_Y[ii] + integrate_energy_Y[ii]

                        integrate_entropy_X[ii] = 10**3 * trapz(trapz(flux_array[0, ...]**2 / prob_ss_array, dx=dx), dx=dx) * timescale
                        integrate_entropy_Y[ii] = 10**3 * trapz(trapz(flux_array[1, ...]**2 / prob_ss_array, dx=dx), dx=dx) * timescale

                        dpx = zeros((N, N))
                        dpy = zeros((N, N))
                        calc_derivative(prob_ss_array, dpx, N, dx, 0)
                        calc_derivative(prob_ss_array, dpy, N, dx, 1)

                        integrate_shannon[ii] = -trapz(trapz(flux_array[0, ...] * dpx / prob_ss_array +
                                                             flux_array[1, ...] * dpy / prob_ss_array, dx=dx), dx=dx) \
                                                * timescale

                        integrate_entropy[ii] = integrate_shannon[ii] + integrate_heat_X[ii] + integrate_heat_Y[ii] - \
                                                integrate_power_X[ii] - integrate_power_Y[ii]

                        force_FoF1 = zeros((N, N))
                        for i in range(N):
                            for j in range(N):
                                force_FoF1[i, j] = -0.5 * Ecouple * sin(positions[i] - positions[j])

                        integrate_couple_X[ii] = trapz(trapz(flux_array[0, ...] * force_FoF1, dx=dx), dx=dx) * timescale
                        integrate_couple_Y[ii] = trapz(trapz(flux_array[1, ...] * (-force_FoF1), dx=dx), dx=dx) * timescale

                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2,
                                                     phase_shift))

                with open(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), "w") as ofile:
                    for j, phase_shift in enumerate(phase_array):
                        ofile.write(
                            f"{phase_shift:.15e}" + "\t"
                            + f"{integrate_power_X[j]:.15e}" + "\t"
                            + f"{integrate_power_Y[j]:.15e}" + "\t"
                            + f"{integrate_heat_X[j]:.15e}" + "\t"
                            + f"{integrate_heat_Y[j]:.15e}" + "\t"
                            + f"{integrate_couple_X[j]:.15e}" + "\t"
                            + f"{integrate_couple_Y[j]:.15e}" + "\t"
                            + f"{integrate_entropy_X[j]:.15e}" + "\t"
                            + f"{integrate_entropy_Y[j]:.15e}" + "\t"
                            + f"{integrate_entropy[j]:.15e}" + "\n"
                        )
                    ofile.flush()


def plot_energy_flow(target_dir):
    phase_array = array([0.0])
    psi1_array = array([8.0])
    psi2_array = array([-7.11, -6.4, -5.33])
    barrier_height = array([2.0])

    input_file_name = (target_dir + "data/200915_energyflows/" + "E0_{0}_E1_{1}/" + "n1_{4}_n2_{5}/" +
                       "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Energy_flow_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_log_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            plt.figure()
            f, ax = plt.subplots(1, 1)
            for j, E0 in enumerate(barrier_height):
                E1 = E0
                if E0 == 0.0:
                    Ecouple_array_total = array([2.0, 4.0, 8.0, 16.0, 32.0, 128.0])
                else:
                    Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))

                power_x = empty(Ecouple_array_total.size)
                power_y = empty(Ecouple_array_total.size)
                heat_x = empty(Ecouple_array_total.size)
                heat_y = empty(Ecouple_array_total.size)
                energy_xy = empty(Ecouple_array_total.size)
                learning_rate = empty(Ecouple_array_total.size)

                for i, Ecouple in enumerate(Ecouple_array_total):
                    try:
                        data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                        power_x[i] = data_array[1]
                        power_y[i] = data_array[2]
                        heat_x[i] = data_array[3]
                        heat_y[i] = data_array[4]
                        energy_xy[i] = data_array[5]
                        learning_rate[i] = data_array[6]
                    except OSError:
                        print('Missing file')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                # print(learning_rate)

                ax.axhline(0, color='black')
                # ax.axhline(1, color='grey')

                ax.plot(Ecouple_array_total, power_x, '-o', label=r'$\beta P_{\rm H^+}$', color='tab:blue')
                ax.plot(Ecouple_array_total, -power_y, '-o', label=r'$-\beta P_{\rm ATP}$', color='tab:orange')
                ax.plot(Ecouple_array_total, heat_x, '-o', label=r'$\dot{Q}_{\rm o}$', color='tab:green')
                ax.plot(Ecouple_array_total, heat_y, '-o', label=r'$\dot{Q}_1$', color='tab:red')
                ax.plot(Ecouple_array_total, -energy_xy, '-o', label=r'$\dot{E}_{\rm o \to 1}$', color='tab:purple')
                # ax.plot(Ecouple_array_total, -energy_xy - learning_rate, '-o',
                #         label=r'$\beta \dot{E}_{\rm o \to 1} - \ell_{\rm o \to 1}$', color='tab:grey')
                # ax.plot(Ecouple_array_total, heat_x + power_x, '-o',
                #         label=r'$J_{\rm o} \partial_{\theta_{\rm o}} V$', color='tab:olive')
                # ax.plot(Ecouple_array_total, -heat_y - power_y, '-o',
                #         label=r'$-J_1 \partial_{\theta_{\rm 1}} V$', color='tab:cyan')
            # ax.set_ylim((0, None))

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xscale('log')
            # ax.set_yscale('log')
            ax.set_xlabel(r'$E_{\rm couple}$', fontsize=14)
            # ax.set_ylabel(r'$\ell_{\rm o \to 1}$', fontsize=14)
            # ax.set_ylabel(r'$\dot{Q}_1 / \dot{E}_{\rm o \to 1}$', fontsize=14)
            # ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
            ax.tick_params(axis='both', labelsize=14)
            ax.yaxis.offsetText.set_fontsize(14)
            # ax.legend(fontsize=12, frameon=False, ncol=1, title=r'$E_{\rm o} = E_1$')
            ax.legend(fontsize=12, frameon=False, ncol=1)

            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_2D_prob():
    output_file_name1 = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "Step_X-Pss_2D_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    # Ecouple_array = array([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0])

    plt.figure()
    f1, ax1 = plt.subplots(1, Ecouple_array.size, figsize=(2.5 * Ecouple_array.size, 3))

    # Find max prob. to set plot range
    # input_file_name = (
    #         "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
    #         "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    # try:
    #     data_array = loadtxt(
    #         input_file_name.format(E0, 128.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
    #     N = int(sqrt(len(data_array)))
    #     prob_ss_array = data_array.reshape((N, N))
    # except OSError:
    #     print('Missing file')
    #     print(input_file_name.format(E0, 128.0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
    #
    # prob_max = amax(prob_ss_array)

    # plots
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                usecols=(0, 2, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array[:, 0])))  # check grid size
            potential_at_pos = data_array[:, 1].reshape((N, N))
            prob_ss_array = data_array[:, 0].reshape((N, N))
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            # integrate using axis=1 integrates out the y component, gives us P(x)
            # prob_ss_x = trapz(prob_ss_array, axis=1)
            # prob_ss_y = trapz(prob_ss_array, axis=0)
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        # flux_array = zeros((2, N, N))
        # calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        # flux_array = asarray(flux_array) / (dx * dx)

        # force_FoF1 = zeros((N, N))
        # force_Fo = zeros((N, N))
        # for i in range(N):
        #     force_Fo[i, :] = -1.5 * E0 * sin(3*positions[i])
        #     for j in range(N):
        #         force_FoF1[i, j] = -0.5 * Ecouple * sin(positions[i] - positions[j])

        step_X = zeros((N, N))
        step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, 5e-2)

        ax1[ii].contourf(step_X - prob_ss_array)

        if ii == 0:
            ax1[ii].set_title(r"$E_{\rm couple}$" + "={}".format(Ecouple))
            ax1[ii].set_ylabel(r'$\theta_{\rm o}$')
            ax1[ii].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
        else:
            ax1[ii].set_title("{}".format(Ecouple))
            ax1[ii].set_yticklabels(['', '', '', '', '', '', ''])
        ax1[ii].set_xlabel(r'$\theta_1$')
        ax1[ii].spines['right'].set_visible(False)
        ax1[ii].spines['top'].set_visible(False)
        # ax1[ii].set_xticks([0, 60, 120, 180, 240, 300, 360])
        # ax1[ii].set_yticks([0, 60, 120, 180, 240, 300, 360])
        ax1[ii].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

    f1.tight_layout()
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_2D_prob_flux():
    output_file_name1 = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "Integrand_LR_scaled_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array_tot = array([8.0, 10.0, 12.0, 14.0, 16.0, 22.0, 24.0, 32.0])

    plt.figure()
    f1, ax1 = plt.subplots(1, Ecouple_array_tot.size, figsize=(3.5*Ecouple_array_tot.size, 3))

    # Find max prob. to set plot range
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, amax(Ecouple_array_tot), E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
            usecols=(0, 2, 3, 4, 5, 6, 7, 8))
        N = int(sqrt(len(data_array)))
        dx = 2 * math.pi / N  # spacing between gridpoints
        positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
        potential_at_pos = data_array[:, 1].reshape((N, N))
        prob_ss_array = data_array[:, 0].reshape((N, N))
        drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
        diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

        flux_array = zeros((2, N, N))
        calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        flux_array = asarray(flux_array) / (dx * dx)

        dflux_array = empty((2, N, N))
        derivative_flux(flux_array, dflux_array, N, dx)

        for i in range(N):
            for j in range(N):
                if prob_ss_array[i, j] == 0:
                    prob_ss_array[i, j] = 10e-18

        learning_rate = -dflux_array[1, ...] * log(prob_ss_array)
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, amax(Ecouple_array_tot), E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

    prob_max = amax(learning_rate)
    prob_min = amin(learning_rate)
    if abs(prob_min) > prob_max:
        prob_max = -prob_min

    # plots
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        if Ecouple in Ecouple_array_peak:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT" +
                    "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        else:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                    "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                usecols=(0, 2, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array[:, 0])))  # check grid size
            dx = 2 * math.pi / N  # spacing between gridpoints
            positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
            potential_at_pos = data_array[:, 1].reshape((N, N))
            prob_ss_array = data_array[:, 0].reshape((N, N))
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        flux_array = zeros((2, N, N))
        calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        flux_array = asarray(flux_array) / (dx * dx)

        dflux_array = empty((2, N, N))
        derivative_flux(flux_array, dflux_array, N, dx)

        for i in range(N):
            for j in range(N):
                if prob_ss_array[i, j] == 0:
                    prob_ss_array[i, j] = 10e-18

        learning_rate = -dflux_array[1, ...] * log(prob_ss_array)

        cs = ax1[ii].contourf(learning_rate.T, linspace(-prob_max, prob_max, 200), cmap=plt.cm.coolwarm,
                              norm=mc.Normalize(vmin=-prob_max, vmax=prob_max))

        # select fewer arrows to draw
        # M = 18  # number of arrows in a row/ column, preferably a number such that N/M is an integer.
        # fluxX = empty((M, M))
        # fluxY = empty((M, M))
        # for k in range(M):
        #     fluxX[k] = flux_array[0, ...][int(N / M) * k, ::int(N / M)]
        #     fluxY[k] = flux_array[1, ...][int(N / M) * k, ::int(N / M)]
        #
        # ax1[ii].quiver(positions[::int(N / M)]*(N/6), positions[::int(N / M)]*(N/6), fluxX.T, fluxY.T, units='xy',
        #                angles='xy', scale_units='xy')

        if ii == 0:
            ax1[ii].set_title(r"$E_{\rm couple}$" + "={}".format(Ecouple))
            ax1[ii].set_ylabel(r'$\theta_{\rm 1}$')
            ax1[ii].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
        else:
            ax1[ii].set_title("{}".format(Ecouple))
            ax1[ii].set_yticklabels(['', '', '', '', '', '', ''])
        ax1[ii].set_xlabel(r'$\theta_{\rm o}$')
        ax1[ii].spines['right'].set_visible(False)
        ax1[ii].spines['top'].set_visible(False)
        # ax1[ii].set_xticks([0, 60, 120, 180, 240, 300, 360])
        # ax1[ii].set_yticks([0, 60, 120, 180, 240, 300, 360])
        ax1[ii].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

    cax = f1.add_axes([0.92, 0.1, 0.01, 0.85])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1, ticks=[-4e-3, -2e-3, 0, 2e-3, 4e-3]
    )
    cbar.set_label(r'$\log \left( P^{\rm ss} \right) \partial_{\theta_1} J_1$', fontsize=16)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    # cbar.ax1.tick_params(labelsize=14)
    # cbar.ax1.yaxis.offsetText.set_fontsize(14)
    cbar.update_ticks()

    # f1.tight_layout()
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_marginal_prob():
    output_file_name1 = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "Pcond_4_difference_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_theta_{6}" + "_.pdf")

    angles = [0, 1]  # index from 0 to 359 gives the position of \theta_o

    Ecouple_array_select = array([2.0, 10.0, 16.0, 22.63, 128.0])
    plt.figure()
    f1, ax1 = plt.subplots(1, Ecouple_array_select.size, figsize=(3 * Ecouple_array_select.size, 3), sharey='all')

    for j in angles:
        print(j)

        ##plots
        for ii, Ecouple in enumerate(Ecouple_array_select):
            # if Ecouple in Ecouple_array_peak:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip" +
                    "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            # else:
            #     input_file_name = (
            #             "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT" +
            #             "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                    usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N
                positions = linspace(0, 2 * math.pi - dx, N)
                prob_ss_array = data_array[:, 0].reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
                # # integrate using axis=1 integrates out the y component, gives us P(x)
                # prob_ss_x = trapz(prob_ss_array, axis=1, dx=1/dx)
                # prob_ss_y = trapz(prob_ss_array, axis=0, dx=1/dx)
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

            # flux_array = zeros((2, N, N))
            # calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N)
            # flux_array = asarray(flux_array) / (dx * dx)

            step_X = empty((N, N))
            step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, 5e-2)

            ax1[ii].axhline(0, color='black')
            ax1[ii].plot(positions, roll(prob_ss_array[int(j*N/3), :]/prob_ss_array.sum(axis=0) -
                                         step_X[int(j*N/3), :]/step_X.sum(axis=0), -int((j*N/3))))

            # ax1[ii].plot(positions, step_X[j, :]/step_X.sum(axis=0), '--')

            if ii == 0:
                ax1[ii].set_title(r"$E_{\rm couple}$" + "={}".format(Ecouple))
                ax1[ii].set_ylabel(r'$P(\theta_{\rm 1} | \theta_{\rm o} = %.1f)$' % (j * 2 * pi / 360))
            else:
                ax1[ii].set_title("{}".format(Ecouple))
            ax1[ii].set_xlabel(r'$\theta_{\rm 1}$')
            ax1[ii].spines['right'].set_visible(False)
            ax1[ii].spines['top'].set_visible(False)
            ax1[ii].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax1[ii].set_ylim([-3e-6, 3e-6])
            ax1[ii].set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi])
            ax1[ii].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
            # ax1[ii].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

    f1.tight_layout()
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, j))


def plot_derivative_flux():
    output_file_name1 = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "Pcond_slice_rotate_2_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    plt.figure()
    f1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    Ecouple_array = array([16.0])

    # ax1.axhline(0, color='black')

    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")

    ##plots
    for ii, Ecouple in enumerate(Ecouple_array):
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                usecols=(0, 1, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array)))  # check grid size
            dx = 2 * math.pi / N
            positions = linspace(0, 2 * math.pi - dx, N)
            prob_ss_array = data_array[:, 0].reshape((N, N))
            prob_eq_array = data_array[:, 1].reshape((N, N))
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        # flux_array = zeros((2, N, N))
        # calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        #
        # dflux_array = zeros((2, N, N))
        # derivative_flux(flux_array, dflux_array, N, dx)
    offset = 0
    ax1.plot(positions, roll(prob_ss_array[offset, :] / prob_ss_array.sum(axis=0), offset), label='$0$')
    ax1.plot(positions, roll(prob_ss_array[offset + int(N/3), :] / prob_ss_array.sum(axis=0), offset + int(2*N/3)), '--', label='$1/3$')
    ax1.plot(positions, roll(prob_ss_array[offset + int(2*N/3), :] / prob_ss_array.sum(axis=0), offset + int(N/3)), '-.', label='$2/3$')

    ax1.set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
    ax1.set_xticklabels(['$0$', '', '$1/3$', '', '$2/3$', '', '$1$'])
    ax1.set_xlim((0, 2*pi))
    ax1.set_ylim((0, None))

    ax1.set_xlabel(r'$\theta_{\rm 1} $', fontsize=12)
    ax1.set_ylabel(r'$P_{\rm ss} (\theta_1 | \theta_{\rm o})$', fontsize=12)
    ax1.yaxis.offsetText.set_fontsize(12)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax1.spines['bottom'].set_visible(False)

    leg = ax1.legend(fontsize=12, loc='best', frameon=False, title=r'Slice at $\theta_o$')
    leg_title = leg.get_title()
    leg_title.set_fontsize(12)

    f1.tight_layout()
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_1D_flux():
    Ecouple_array = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])

    plt.figure()
    f, ax = plt.subplots(1, Ecouple_array.size, figsize=(18, 3), sharey='all')

    for i, Ecouple in enumerate(Ecouple_array):
        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200810_bipartite/" +
                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                           "_outfile.dat")

        output_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
                            "Flux_flat_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                            "_outfile.pdf")
        try:
            data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                                 usecols=(0, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array)))  # check grid size
            dx = 2 * math.pi / N  # spacing between gridpoints
            positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints

            prob_ss_array = data_array[:, 0].reshape((N, N))
            drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        flux_array = zeros((2, N, N))
        calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

        flux_x = trapz(flux_array[0, ...], axis=1)
        flux_y = trapz(flux_array[1, ...], axis=0)

        ax[i].plot(positions, flux_x)
        ax[i].plot(positions, flux_y)


        if i == 0:
            ax[i].set_title("$E_{couple}$" + "={}".format(Ecouple))
            ax[i].set_ylabel('$J_i(\\theta_{\\rm 1})$')
        else:
            ax[i].set_title("{}".format(Ecouple))
        ax[i].set_xlabel('$\\theta_{\\rm i}$')
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
        ax[i].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))


if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    # flux_power_efficiency(target_dir)
    # heat_work_info(target_dir)
    # plot_power_Ecouple(target_dir)
    # plot_power_efficiency_Ecouple(target_dir)
    # plot_power_Ecouple_grid(target_dir)
    # plot_power_efficiency_phi(target_dir)
    # plot_power_phi_single(target_dir)
    # plot_nn_power_efficiency_Ecouple(target_dir)
    # plot_nn_power_efficiency_phi(target_dir)
    # plot_n0_power_efficiency_Ecouple(target_dir)
    # calc_heat_flow()
    plot_energy_flow(target_dir)
    # plot_2D_prob()
    # plot_2D_prob_flux()
    # plot_marginal_prob()
    # plot_derivative_flux()
    # plot_1D_flux()
