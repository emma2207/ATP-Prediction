from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz, log, sin, amax, \
    concatenate, sort, exp, ones, roll, average
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from utilities import step_probability_X
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 720  # N x N grid is used for Fokker-Planck simulations
dx = 2 * math.pi / N  # spacing between gridpoints
positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
timescale = 1.5 * 10**4  # conversion factor between simulation and experimental timescale

E0 = 2.0  # barrier height Fo
E1 = 2.0  # barrier height F1
psi_1 = 4.0  # chemical driving force on Fo
psi_2 = -2.0  # chemical driving force on F1
num_minima1 = 12.0  # number of barriers in Fo's landscape
num_minima2 = 12.0  # number of barriers in F1's landscape

min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])  # number of energy minima/ barriers

Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])  # coupling strengths
Ecouple_array_zero = array([0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 128.0])
Ecouple_array_peak = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])
Ecouple_array_double = array([2.83, 5.66, 11.31, 22.63, 45.25, 90.51])
Ecouple_array_double2 = array([11.31, 22.63, 45.25, 90.51])
Ecouple_array_quad = array([1.19, 1.68, 2.38, 3.36, 4.76, 6.73, 9.51, 13.45, 19.03, 26.91, 38.05, 53.82, 76.11, 107.63])

Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))


def calc_flux_2(p_now, drift_at_pos, diffusion_at_pos, flux_array, N, dx):
    # calculates the local flux from simulation output
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
    # calculates a derivative wrt of the flux wrt to each variable
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
    # calculates a derivative of a 2D quantity
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


def heat_work_info(target_dir):
    # data processing of raw simulation data into averaged quantities
    Ecouple_array_tot = array([32.0, 45.25, 64.0, 90.51, 128.0, 181.0, 256.0, 362.0, 512.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    phase_array = array([0.0])

    output_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")

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
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/210818_collapsed_curve_spectral/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

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


def plot_power_efficiency_Ecouple_hor(target_dir):
    # methods chapter. code testing plots
    # Energy chapter. comparing different barrier heights
    # plot power and efficiency vs coupling strength
    clr = ['C1']

    barrier_heights = array([2.0])
    barrier_height_labels = [r'$\rm Fokker \mbox{-} Planck$']

    output_file_name = (target_dir + "results/" + "P_ATP_eff_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" +
                        "_.pdf")
    f, axarr = plt.subplots(1, 2, sharex='all', sharey='none', figsize=(10, 4))

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis
    maxpower = 2 * pi * 0.000085247 * timescale  # max power for E0=E1=2
    axarr[0].axhline(maxpower, linestyle=':', color=clr[0], linewidth=2, label=r'$\rm semi \mbox{-} analytical$')
    # maxpower = 30.  # maxpower for E0=E1=0
    # axarr[0].axhline(maxpower, linestyle=':', color=clr[0], linewidth=2, label=r'$\rm exact$')

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x axis
    axarr[1].axhline(1, linestyle=':', color=clr[0], linewidth=2) #max efficiency

    # zero-barrier results
    input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
                       "plotting_data/Driving_forces/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    Ecouple_array2 = array(data_array[:, 0])
    flux_x_array = array(data_array[:, 1])
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    # axarr[0].plot(Ecouple_array2, 2 * pi * power_y * timescale, '-', linewidth=2, label=r'$\rm semi \mbox{-} analytical$')
    # axarr[1].plot(Ecouple_array2, flux_y_array / flux_x_array, '-', linewidth=2)

    # # Fokker-Planck results
    for j, E0 in enumerate(barrier_heights):
        E1 = E0
        power_x_array = []
        power_y_array = []
        eff_array = []

        if E0 == 0.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        elif E0 == 2.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))
        else:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (target_dir + "/data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                               "power_heat_info_"
                               + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(1, 2))
                power_x_array = append(power_x_array, data_array[0])
                power_y_array = append(power_y_array, data_array[1])
                eff_array = append(eff_array, data_array[1]/data_array[0])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        if E0 > 0:
            axarr[0].plot(Ecouple_array_tot, -power_y_array, '-', color=clr[j],
                          markersize=8)
            axarr[1].plot(Ecouple_array_tot, eff_array / (psi_2 / psi_1), '-', markersize=8, color=clr[j])

        axarr[0].plot(Ecouple_array_tot, -power_y_array, 'o', label=barrier_height_labels[j], color=clr[j],
                      markersize=8)
        axarr[1].plot(Ecouple_array_tot, eff_array / (psi_2 / psi_1), 'o', markersize=8, color=clr[j])

    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r'$-\beta \mathcal{P}_{\rm ATP} \ (\rm s^{-1}) $', fontsize=18)
    axarr[0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=18)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].spines['bottom'].set_visible(False)
    axarr[0].set_xlim((1.7, 140))
    axarr[0].set_ylim((-42, 18))
    axarr[0].tick_params(axis='both', labelsize=14)
    # axarr[0].set_yscale('log')

    # leg = axarr[0].legend(fontsize=14, loc='lower right', frameon=False, title=r'$\beta E_{\rm o} = \beta E_1$')
    leg = axarr[0].legend(fontsize=14, loc='lower right', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    axarr[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=18)
    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=18)
    axarr[1].set_xscale('log')
    axarr[1].set_xlim((1.7, 140))
    axarr[1].set_ylim((-0.52, 1.05))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].spines['bottom'].set_visible(False)
    axarr[1].set_yticks([-0.5, 0, 0.5, 1.0])
    axarr[1].tick_params(axis='both', labelsize=14)

    f.text(0.02, 0.92, r'$\rm{a)}$', ha='center', fontsize=18)
    f.text(0.52, 0.92, r'$\rm{b)}$', ha='center', fontsize=18)
    f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_power_efficiency_Ecouple_ver(target_dir):
    # Energy chapter. comparing different barrier heights
    # plot power and efficiency vs coupling strength
    clr = ['C1', 'C9']

    barrier_heights = array([2.0, 4.0])
    barrier_height_labels = [r'$2$', r'$4$']

    output_file_name = (target_dir + "results/" + "P_ATP_eff_Ecouple_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" +
                        "_.pdf")
    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(5.5, 7))

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis
    # maxpower = 2 * pi * 0.000085247 * timescale  # max power for E0=E1=2
    # maxpower = 30.  # maxpower for E0=E1=0
    # axarr[0].axhline(maxpower, linestyle=':', color=clr, linewidth=2, label=r'$\rm semi \mbox{-} analytical$')

    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x axis
    # axarr[1].axhline(1, linestyle=':', color=clr, linewidth=2) #max efficiency

    # zero-barrier results
    input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
                       "plotting_data/Driving_forces/" + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    Ecouple_array2 = array(data_array[:, 0])
    flux_x_array = array(data_array[:, 1])
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', linewidth=2, label=r'$0$')
    axarr[1].plot(Ecouple_array2, flux_y_array / flux_x_array, '-', linewidth=2)

    # # Fokker-Planck results
    for j, E0 in enumerate(barrier_heights):
        E1 = E0
        power_x_array = []
        power_y_array = []
        eff_array = []

        if E0 == 0.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        elif E0 == 2.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))
        else:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (target_dir + "/data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                               "power_heat_info_"
                               + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(1, 2))
                power_x_array = append(power_x_array, data_array[0])
                power_y_array = append(power_y_array, data_array[1])
                eff_array = append(eff_array, data_array[1]/data_array[0])
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        if E0 > 0:
            idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y_array.argmin()])).argmin()
            print(Ecouple_array_tot[idx - 1], eff_array[idx - 1]/-0.5, Ecouple_array_tot[idx + 1], eff_array[idx + 1]/-0.5)
            axarr[0].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], -10 ** 2, 10 ** 2,
                                  facecolor=clr[j], alpha=0.4)
            axarr[1].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], -10 ** 2, 10 ** 2,
                                  facecolor=clr[j], alpha=0.4)
            axarr[0].plot(Ecouple_array_tot, -power_y_array, '-', color=clr[j],
                          markersize=8)
            axarr[1].plot(Ecouple_array_tot, eff_array / (psi_2 / psi_1), '-', markersize=8, color=clr[j])

        axarr[0].plot(Ecouple_array_tot, -power_y_array, 'o', label=barrier_height_labels[j], color=clr[j],
                      markersize=8)
        axarr[1].plot(Ecouple_array_tot, eff_array / (psi_2 / psi_1), 'o', markersize=8, color=clr[j])

    axarr[0].yaxis.offsetText.set_fontsize(14)
    axarr[0].tick_params(axis='y', labelsize=14)
    axarr[0].set_ylabel(r'$-\beta \mathcal{P}_{\rm ATP} \ (\rm s^{-1}) $', fontsize=18)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].spines['bottom'].set_visible(False)
    axarr[0].set_xlim((1.7, 140))
    axarr[0].set_ylim((-60, 32))
    axarr[0].tick_params(axis='both', labelsize=14)
    # axarr[0].set_yscale('log')

    leg = axarr[0].legend(fontsize=14, loc='lower right', frameon=False, title=r'$\beta E_{\rm o} = \beta E_1$')
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    axarr[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=18)
    axarr[1].set_ylabel(r'$\eta / \eta^{\rm max}$', fontsize=18)
    axarr[1].set_xscale('log')
    axarr[1].set_xlim((1.7, 140))
    axarr[1].set_ylim((-0.52, 1.05))
    axarr[1].spines['right'].set_visible(False)
    axarr[1].spines['top'].set_visible(False)
    axarr[1].spines['bottom'].set_visible(False)
    axarr[1].set_yticks([-0.5, 0, 0.5, 1.0])
    axarr[1].tick_params(axis='both', labelsize=14)

    f.text(0.04, 0.95, r'$\rm{a)}$', ha='center', fontsize=18)
    f.text(0.04, 0.49, r'$\rm{b)}$', ha='center', fontsize=18)
    f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_2D_prob_flux_thesis():
    # Energy chapter. steady state probability and flux plots
    # 3x2 plots of local pss (heatmap) and J (arrows) for a few coupling strengths
    output_file_name1 = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" +
            "Pss_flux_2D_scaled_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array_tot = array([0.0, 4.0, 8.0, 16.0, 32.0, 128.0])
    Ecouplelst = ['$0$', '$4$', '$8$', '$16$', '$32$', '$128$']

    plt.figure()
    f1, ax1 = plt.subplots(2, 3, figsize=(6, 4))

    # Find max prob. to set plot range
    if E0 == 2.0:
        input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    else:
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
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
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, amax(Ecouple_array_tot), E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

    prob_max = amax(prob_ss_array)

    # plots
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        if E0 == 2.0:
            if Ecouple in Ecouple_array_peak:
                input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra" +
                        "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
            else:
                input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                        "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        else:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
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
        print(ii, ii//3, ii%3)
        cs = ax1[ii // 3, ii % 3].contourf(prob_ss_array.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)

        # select fewer arrows to draw
        M = 15  # number of arrows in a row/ column, preferably a number such that N/M is an integer.
        fluxX = empty((M, M))
        fluxY = empty((M, M))

        for k in range(M):
            for l in range(M):
                fluxX[k, l] = average(flux_array[0, ...][(N//M * k):(N//M * (k + 1)), (N//M * l):(N//M * (l + 1))])
                fluxY[k, l] = average(flux_array[1, ...][(N//M * k):(N//M * (k + 1)), (N//M * l):(N//M * (l + 1))])

        ax1[ii // 3, ii % 3].quiver(positions[::(N // M)]*(N/6), positions[::(N // M)]*(N/6), fluxX.T, fluxY.T,
                       units='xy', angles='xy', scale_units='xy', pivot='mid')

        if ii % 3 == 0:
            ax1[ii // 3, ii % 3].set_ylabel(r'$\theta_{\rm 1}$', fontsize=14)
            ax1[ii // 3, ii % 3].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
        else:
            ax1[ii // 3, ii % 3].set_yticklabels(['', '', '', '', '', '', ''])
        if ii == 0:
            ax1[ii // 3, ii % 3].set_title(r"$\beta E_{\rm couple}$" + "={}".format(Ecouplelst[ii]))
        else:
            ax1[ii // 3, ii % 3].set_title("{}".format(Ecouplelst[ii]))
        if ii < 3:
            ax1[ii // 3, ii % 3].set_xticklabels(['', '', '', '', '', '', ''])
        else:
            ax1[ii // 3, ii % 3].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
            ax1[ii // 3, ii % 3].set_xlabel(r'$\theta_{\rm o}$', fontsize=14)

        ax1[ii // 3, ii % 3].spines['right'].set_visible(False)
        ax1[ii // 3, ii % 3].spines['top'].set_visible(False)
        ax1[ii // 3, ii % 3].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii // 3, ii % 3].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])

    cax = f1.add_axes([0.94, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1
    )
    cbar.set_label(r'$p_{\rm ss}(\theta_{\rm o}, \theta_1)$', fontsize=14)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplots_adjust(hspace=0.3)

    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_energy_flow(target_dir):
    # Energy chapter. energy flows vs coupling strength
    # input power, output power, heat flows X and Y, power from X to Y
    phase_array = array([0.0])
    psi1_array = array([8.0])
    psi2_array = array([-4.0])
    barrier_height = array([2.0])
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Energy_flow_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for k, phi in enumerate(phase_array):
                plt.figure()
                f, ax = plt.subplots(1, 1)
                for j, E0 in enumerate(barrier_height):
                    E1 = E0
                    if E0 == 0.0:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
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

                    ax.axhline(0, color='black')

                    ax.plot(Ecouple_array_total, power_x, '-o', label=r'$\mathcal{P}_{\rm H^+}$', color='tab:blue')
                    ax.plot(Ecouple_array_total, power_y, '-o', label=r'$\mathcal{P}_{\rm ATP}$', color='tab:orange')
                    ax.plot(Ecouple_array_total, heat_x, '-o', label=r'$\dot{Q}_{\rm o}$', color='tab:green')
                    ax.plot(Ecouple_array_total, heat_y, '-o', label=r'$\dot{Q}_1$', color='tab:red')
                    ax.plot(Ecouple_array_total, -energy_xy, '-o', label=r'$\mathcal{P}_{\rm o \to 1}$', color='tab:purple')

                ax.set_ylim((-120, 120))
                ax.set_xlim((2, None))

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_xscale('log')
                # ax.set_yscale('log')
                ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
                ax.set_ylabel(r'$\textrm{Energy flow} \ (k_{\rm B}T \cdot \rm s^{-1})$', fontsize=16)
                ax.tick_params(axis='both', labelsize=14)
                ax.yaxis.offsetText.set_fontsize(14)
                # ax.legend(fontsize=14, frameon=False, ncol=1, loc='best')

                f.tight_layout()
                f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_power_Ecouple_grid(target_dir):
    # Energy chapter. grid of plots 3x3
    # output power vs coupling strength for different driving forces
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    barrier_heights = [2.0, 4.0]
    colorlst = ['C1', 'C9']

    output_file_name = (target_dir + "results/" + "P_ATP_Ecouple_grid_solid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # line at highest Ecouple power
            # input_file_name = (
            #                 "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
            #                 "plotting_data/Driving_forces/" +
            #                 "Power_Ecouple_inf_grid_E0_2.0_E1_2.0_n1_3.0_n2_3.0_outfile.dat")
            # try:
            #     inf_array = loadtxt(input_file_name, usecols=2)
            # except OSError:
            #     print('Missing file Infinite Power Coupling')
            #
            # axarr[i, j].axhline(2*pi*inf_array[i*3 + j] * timescale, color='grey', linestyle=':', linewidth=1)

            # peak position estimate output power from theory
            Ecouple_est = 3.31 + 4 * pi * (psi_1 - psi_2) / 9

            # for jj, E0 in enumerate(barrier_heights):

            # axarr[i, j].axvline(midpoint, linestyle=(offset[jj], (1, 1)), color=colorlst[jj], linewidth=1 * width,
            #                     alpha=0.4)

            # zero-barrier result
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
                               "plotting_data/Driving_forces/" + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_y_array = array(data_array[:, 2])
            power_y = -flux_y_array * psi_2

            axarr[i, j].plot(Ecouple_array2, 2*pi*power_y*timescale, '-', color='C0', linewidth=3)

            # Fokker-Planck results (barriers)
            for jj, E0 in enumerate(barrier_heights):
                E1 = E0
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array_tot):
                    input_file_name = (
                                target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=2)

                        power_y = array(data_array)
                        power_y_array = append(power_y_array, power_y)
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                # calculate line position and width to include 'error' for peak power
                idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y_array.argmin()])).argmin()
                axarr[i, j].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], 10**(-2), 10**3,
                                         facecolor=colorlst[jj], alpha=0.4)

                axarr[i, j].plot(Ecouple_array_tot, -power_y_array, 'o-', color=colorlst[jj], markersize=8)

            axarr[i, j].axvline(Ecouple_est, color='black', linestyle='--', linewidth=2)

            axarr[i, j].set_xscale('log')
            axarr[i, j].set_yscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((2, 150))
            elif j == 1:
                axarr[i, j].set_xlim((3, 150))
            else:
                axarr[i, j].set_xlim((4, 150))

            if i == 0:
                axarr[i, j].set_ylim((0.03, 10))
                axarr[i, j].set_yticks([0.1, 1, 10])
                # axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            elif i == 1:
                axarr[i, j].set_ylim((0.3, 31))
                axarr[i, j].set_yticks([1, 10])
                # axarr[i, j].set_yticklabels([r'$0$', r'$15$', r'$30$'])
            else:
                axarr[i, j].set_ylim((4, 122))
                axarr[i, j].set_yticks([10, 100])
                # axarr[i, j].set_yticklabels([r'$0$', r'$50$', r'$100$'])

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=16)

            # highlight one plot with red axes
            # if j == 2 and i == 1:
            #     axarr[i, j].tick_params(axis='x', colors='red', which='both')
            #     axarr[i, j].tick_params(axis='y', colors='red', which='both')
            #     axarr[i, j].spines['left'].set_color('red')
            #     axarr[i, j].spines['bottom'].set_color('red')
            # else:
            axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', va='center', rotation='vertical',
           fontsize=20)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=20)
    f.text(0.95, 0.5, r'$\beta \mu_{\rm H^+}\ (\rm rad^{-1})$', va='center', rotation=270, fontsize=20)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_efficiency_Ecouple_grid(target_dir):
    # Energy chapter. grid of plots 3x3
    # efficiency vs coupling strength for different driving forces
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    barrier_heights = [2.0, 4.0]
    colorlst = ['C1', 'C9']
    freq = 2
    offset = [0, freq]

    output_file_name = (target_dir + "results/" + "Efficiency_Ecouple_grid_solid_" + "E0_{0}_E1_{1}_n1_{2}_n2_{3}" + "_.pdf")
    f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(8, 6))
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print('Chemical driving forces:', psi_1, psi_2)

            # rate calculations theory line
            pos = linspace(1, 150, 200)  # array of coupling strengths
            theory = 1 - 3 * exp((pi / 3) * (psi_1 - psi_2) - 0.75 * pos)

            # zero-barrier result
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
                               "plotting_data/Driving_forces/" + "Flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            data_array = loadtxt(input_file_name.format(psi_1, psi_2))
            Ecouple_array2 = array(data_array[:, 0])
            flux_x_array = array(data_array[:, 1])
            flux_y_array = array(data_array[:, 2])

            axarr[i, j].plot(Ecouple_array2, flux_y_array/flux_x_array, '-', color='C0', linewidth=3)

            # Fokker-Planck results (barriers)
            for jj, E0 in enumerate(barrier_heights):
                E1 = E0
                power_x_array = []
                power_y_array = []
                for ii, Ecouple in enumerate(Ecouple_array_tot):
                    input_file_name = (
                                target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    try:
                        # print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                        data_array = loadtxt(
                            input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(1, 2))

                        power_x = array(data_array[0])
                        power_y = array(data_array[1])
                        power_y_array = append(power_y_array, power_y)
                        power_x_array = append(power_x_array, power_x)
                    except OSError:
                        print('Missing file flux')
                        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                # calculate line position and width to include 'error' for peak power
                # idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y_array.argmin()])).argmin()
                # axarr[i, j].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], 10**(-2), 10**3,
                #                          facecolor=colorlst[jj], alpha=0.4)

                axarr[i, j].plot(Ecouple_array_tot, (power_y_array/power_x_array)/(psi_2/psi_1), 'o-', color=colorlst[jj], markersize=8)

            axarr[i, j].plot(pos, theory, color='black', linestyle='--', linewidth=2)

            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_xticks([1., 10., 100.])
            if j == 0:
                axarr[i, j].set_xlim((2, 150))
            elif j == 1:
                axarr[i, j].set_xlim((3, 150))
            else:
                axarr[i, j].set_xlim((4, 150))

            axarr[i, j].set_ylim((0.0, 1.05))
            # axarr[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=16)

            axarr[i, j].tick_params(axis='both', labelsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=20)
    f.text(0.01, 0.5, r'$\eta / \eta^{\rm max}$', va='center', rotation='vertical',
           fontsize=20)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=20)
    f.text(0.95, 0.5, r'$\beta \mu_{\rm H^+}\ (\rm rad^{-1})$', va='center', rotation=270, fontsize=20)

    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_nn_power_efficiency_Ecouple(target_dir):
    # Energy chapter. varying number of barriers
    # power and efficiency vs coupling strength including infinite coupling result
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))

    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    output_file_name = (
            target_dir + "results/" + "P_atp_nn_eff_Ecouple_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}" + "_.pdf")

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                        target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                        + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple),
                    usecols=2)
                power_y = array(data_array)
                power_y_array = append(power_y_array, power_y)
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

        # Infinite coupling data
        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Number_barriers/" +
                           "Power_ATP_Ecouple_inf_no_n1_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 0].plot(Ecouple_array_tot, -power_y_array, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])
        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, linestyle='-',
                         color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].tick_params(axis='y', labelsize=14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} \ (\rm s^{-1}) $', fontsize=20)
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
                    target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=(1, 2))
                eff_array = append(eff_array, data_array[1]/data_array[0])
            except OSError:
                print('Missing file efficiency')

        # infinite coupling value
        axarr[1, 0].plot(Ecouple_array_tot, -eff_array/0.5, marker=markerlst[j], markersize=6, linestyle='-',
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

    leg = axarr[0, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o} = n_1$', fontsize=14,
                             loc=[0.19, 0.02], frameon=False, ncol=1)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.05, 0.92, r'$\rm a)$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\rm b)$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_n0_power_efficiency_Ecouple(target_dir):
    # Energy chapter. varying number of barriers
    # power and efficiency coupling strength including infinite coupling result
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    output_file_name = (
                target_dir + "results/Patp_eff_Ecouple_n0_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n2_{4}" + "_.pdf")
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(6, 6),
                            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [2, 1]})

    # power plot
    # Fokker-Planck results (barriers
    for j, num_min in enumerate(min_array):
        power_y_array = []
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (
                    target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple),
                    usecols=2)
                power_y = array(data_array)
                power_y_array = append(power_y_array, power_y)
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))

        axarr[0, 0].plot(Ecouple_array_tot, -power_y_array, marker=markerlst[j], markersize=6,
                         linestyle='-', color=color_lst[j])

        # infinite coupling result
        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Number_barriers/" +
                           "Power_ATP_Ecouple_inf_no_varies_n1_3.0_E0_2.0_E1_2.0_psi1_4.0_psi2_-2.0_outfile.dat")
        try:
            data_array = loadtxt(input_file_name)
            power_inf = array(data_array[j, 1])
        except OSError:
            print('Missing file infinite coupling power')

        axarr[0, 1].plot([300], 2*pi*power_inf*timescale, marker=markerlst[j], markersize=6, color=color_lst[j])

    axarr[0, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0, 0].yaxis.offsetText.set_fontsize(14)
    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} \ (\rm s^{-1}) $', fontsize=20)
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
                    target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple), usecols=(1, 2))
                eff_array = append(eff_array, data_array[1]/data_array[0])
            except OSError:
                print('Missing file efficiency')

        axarr[1, 0].plot(Ecouple_array_tot, eff_array / (psi_2 / psi_1), marker=markerlst[j], markersize=6,
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

    leg = axarr[0, 0].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14,
                             loc=[0.17, 0.02], frameon=False, ncol=1)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    # f.text(0.5, 0.6, r'$n_1=3$', ha='center', fontsize=14)
    f.text(0.05, 0.92, r'$\rm a)$', ha='center', fontsize=20)
    f.text(0.05, 0.37, r'$\rm b)$', ha='center', fontsize=20)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, 3.0))


def plot_power_phi_single(target_dir):
    # Energy chapter. phase offset
    # power vs phase offset for different coupling strengths
    colorlst = ['C2', 'C3', 'C1', 'C4']
    markerlst = ['D', 's', 'o', 'v']
    Ecouple_array = array([2.0, 8.0, 16.0, 32.0])
    labellst = ['$2$', '$8$', '$16$', '$32$']

    output_file_name = (target_dir
                        + "results/Power_ATP_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(8.5, 4))
    ax.axhline(0, color='black', linewidth=1)

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" +
                           "plotting_data/Phase_offset/" + "flux_power_efficiency_"
                           + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                usecols=(0, 4))
            phase_array = data_array[:, 0]
            power_y_array = data_array[:, 1]

            ax.plot(phase_array, -2*pi*power_y_array*timescale, linestyle='-', marker=markerlst[ii],
                    label=labellst[ii], markersize=8, linewidth=2, color=colorlst[ii])
        except OSError:
            print('Missing file')

    # Infinite coupling result
    input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" + "plotting_data/Phase_offset/"
                       + "Flux_phi_Ecouple_inf_Fx_4.0_Fy_-2.0_test.dat")
    data_array = loadtxt(input_file_name, usecols=(0, 1))
    phase_array = data_array[:, 0]
    power_y_array = -psi_2 * data_array[:, 1]
    cs = ax.plot(phase_array[:61], 2*pi*power_y_array[:61]*timescale, '-', label=f'$\infty$', linewidth=2, color='C6')

    # zero-barrier results
    input_file_name = (
            "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/" + "plotting_data/Driving_forces/"
            + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    ax.axhline(2 * pi * power_y[28] * timescale, color='C0', linewidth=2, label=r'$\infty,\, \rm no \, barriers$')

    ax.tick_params(axis='both', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.set_xlim((0, 2.1))
    ax.set_ylim((-41, 31))
    ax.set_yticks([-40, -20, 0, 20])

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], title=r'$\beta E_{\rm couple}$', fontsize=14,
                    frameon=False, ncol=1, bbox_to_anchor=(1.05, 0.9))
    leg_title = leg.get_title()
    leg_title.set_fontsize(18)

    f.text(0.41, 0.02, r'$n \phi\ (\rm rad)$', fontsize=18, ha='center')
    plt.ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=18)
    plt.xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3],
               ['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    f.tight_layout()
    f.subplots_adjust(bottom=0.14)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2))


def plot_2D_prob_flux_phi_thesis(target_dir):
    output_file_name1 = (
            target_dir + "results/" +
            "Pss_flux_2D_phi_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_.pdf")

    Ecouple = 16.0
    phase_array = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533])
    phaselst = [r'$0$', r'$\pi / 3$', r'$2 \pi / 3$', r'$\pi$', r'$4 \pi / 3$', r'$5 \pi / 3$', r'$2 \pi$']

    plt.figure()
    f1, ax1 = plt.subplots(2, 3, figsize=(6, 4))

    # Find max prob. to set plot range
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_array[0]),
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
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_array[0]))

    prob_max = amax(prob_ss_array)
    print(prob_max)

    # plots
    for ii, phi in enumerate(phase_array):
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
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
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

        flux_array = zeros((2, N, N))
        calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        flux_array = asarray(flux_array) / (dx * dx)

        if ii == 0:
            cs = ax1[ii // 3, ii % 3].contourf(prob_ss_array.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)
        else:
            ax1[ii // 3, ii % 3].contourf(prob_ss_array.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)

        # select fewer arrows to draw
        M = 15  # number of arrows in a row/ column, preferably a number such that N/M is an integer.
        fluxX = empty((M, M))
        fluxY = empty((M, M))
        for k in range(M):
            fluxX[k] = flux_array[0, ...][int(N / M) * k, ::int(N / M)]
            fluxY[k] = flux_array[1, ...][int(N / M) * k, ::int(N / M)]

        ax1[ii // 3, ii % 3].quiver(positions[::int(N / M)]*(N/6), positions[::int(N / M)]*(N/6), fluxX.T, fluxY.T,
                                    units='xy', angles='xy', scale_units='xy')

        if ii % 3 == 0:
            ax1[ii // 3, ii % 3].set_ylabel(r'$\theta_{\rm 1}$', fontsize=14)
            ax1[ii // 3, ii % 3].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
        else:
            ax1[ii // 3, ii % 3].set_yticklabels(['', '', '', '', '', '', ''])
        if ii == 0:
            ax1[ii // 3, ii % 3].set_title(r"$n \phi$" + "={}".format(phaselst[ii]))
        else:
            ax1[ii // 3, ii % 3].set_title("{}".format(phaselst[ii]))
        if ii < 3:
            ax1[ii // 3, ii % 3].set_xticklabels(['', '', '', '', '', '', ''])
        else:
            ax1[ii // 3, ii % 3].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
            ax1[ii // 3, ii % 3].set_xlabel(r'$\theta_{\rm o}$', fontsize=14)

        ax1[ii // 3, ii % 3].spines['right'].set_visible(False)
        ax1[ii // 3, ii % 3].spines['top'].set_visible(False)
        ax1[ii // 3, ii % 3].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii // 3, ii % 3].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])

    cax = f1.add_axes([0.94, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1
    )
    cbar.set_label(r'$p_{\rm ss}(\theta_{\rm o}, \theta_1)$', fontsize=14)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplots_adjust(hspace=0.3)
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), bbox_inches='tight')


def plot_nn_power_efficiency_phi(target_dir):
    # Energy chapter. phase offset
    # power and efficiency vs coupling strength
    phase_array = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599, 6.28319])
    Ecouple_array = array([16.0])
    n_labels = ['$1$', '$2$', '$3$', '$6$', '$12$']
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']

    f, axarr = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(6, 4.5))

    output_file_name = (
            target_dir + "results/power_efficiency_phi_vary_n_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_Ecouple_{4}" + "_log_.pdf")

    # power plot
    axarr[0].axhline(0, color='black', linewidth=1)  # x-axis

    # zero-barrier results
    input_file_name = (
                "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Driving_forces/"
                + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
    data_array = loadtxt(input_file_name.format(psi_1, psi_2))
    flux_y_array = array(data_array[:, 2])
    power_y = -flux_y_array * psi_2
    axarr[0].axhline(2*pi*power_y[28]*timescale, color='C0', linewidth=2, label='$0$')

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        if num_min != 3.0:
            input_file_name = (
                        "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Number_barriers/"
                        + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        else:
            input_file_name = (
                    "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Phase_offset/"
                    + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        for ii, Ecouple in enumerate(Ecouple_array):
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=4)
                if num_min != 3.0:
                    power_y = data_array
                else:
                    power_y = data_array[::2]
            except OSError:
                print('Missing file flux')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        if num_min != 3.0:
            power_y = append(power_y, power_y[0])
        axarr[0].plot(phase_array, -2*pi*power_y*timescale, '-', markersize=8, color=color_lst[i], marker=markerlst[i],
                      label=n_labels[i])

    axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr[0].set_xticks([0, pi/9, 2*pi/9, pi/3, 4*pi/9, 5*pi/9, 2*pi/3])
    axarr[0].yaxis.offsetText.set_fontsize(16)
    axarr[0].tick_params(axis='both', labelsize=16)
    axarr[0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP}\ (\rm s^{-1})$', fontsize=20)
    axarr[0].spines['right'].set_visible(False)
    axarr[0].spines['top'].set_visible(False)
    axarr[0].set_ylim((0, None))
    axarr[0].set_xlim((0, 6.3))

    leg = axarr[0].legend(title=r'$n_{\rm o} = n_1$', fontsize=16, loc='lower center', frameon=False, ncol=1,
                          bbox_to_anchor=(1.15, -0.75))
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    #########################################################
    # efficiency plot
    axarr[1].axhline(0, color='black', linewidth=1)  # x-axis
    axarr[1].axhline(1, color='C0', linewidth=2, label='$0$')  # max efficiency
    axarr[1].set_aspect(1.5)

    # Fokker-Planck results (barriers)
    for i, num_min in enumerate(min_array):
        if num_min != 3.0:
            input_file_name = (
                        "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Number_barriers/"
                        + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        else:
            input_file_name = (
                    "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/Phase_offset/"
                    + "flux_power_efficiency_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
        for ii, Ecouple in enumerate(Ecouple_array):
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple), usecols=5)
                if num_min != 3.0:
                    eff_array = data_array
                else:
                    eff_array = data_array[::2]
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
    axarr[1].yaxis.offsetText.set_fontsize(16)
    axarr[1].tick_params(axis='both', labelsize=16)
    axarr[1].set_yticks([0, 0.5, 1.0])
    axarr[1].set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/3, 2*pi])
    axarr[1].set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    f.text(0.55, 0.01, r'$n \phi \ (\rm rad)$', fontsize=20, ha='center')  # xlabel seems broken
    f.text(0.03, 0.93, r'$\rm a)$', fontsize=20)
    f.text(0.03, 0.4, r'$\rm b)$', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_energy_flow_zoom(target_dir):
    # Energy chapter. energy flows vs coupling strength
    # input power, output power, heat flows X and Y, power from X to Y
    phi = 0.0
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Energy_flow_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_zoom_.pdf")

    plt.figure()
    f, ax = plt.subplots(2, 1, sharex='all', figsize=(5, 7))
    for j in range(2):
        E1 = E0
        if E0 == 0.0:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
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

        ax[j].axhline(0, color='black')

        ax[j].plot(Ecouple_array_total, power_x, '-o', label=r'$\mathcal{P}_{\rm H^+}$', color='tab:blue')
        ax[j].plot(Ecouple_array_total, power_y, '-o', label=r'$\mathcal{P}_{\rm ATP}$', color='tab:orange')
        ax[j].plot(Ecouple_array_total, heat_x, '-o', label=r'$\dot{Q}_{\rm o}$', color='tab:green')
        ax[j].plot(Ecouple_array_total, heat_y, '-o', label=r'$\dot{Q}_1$', color='tab:red')
        ax[j].plot(Ecouple_array_total, -energy_xy, '-o', label=r'$\mathcal{P}_{\rm o \to 1}$', color='tab:purple')

        if j == 0:
            ax[j].set_ylim((-700, 700))
        else:
            ax[j].set_ylim((-120, 120))
            ax[j].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)

        ax[j].set_xlim((2, None))

        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['bottom'].set_visible(False)
        ax[j].set_xscale('log')
        # ax[j].set_yscale('log')
        ax[j].set_ylabel(r'$\textrm{Energy flow} \ (k_{\rm B}T \cdot \rm s^{-1})$', fontsize=16)
        ax[j].tick_params(axis='both', labelsize=14)
        ax[j].yaxis.offsetText.set_fontsize(14)
        # ax.legend(fontsize=14, frameon=False, ncol=1, loc='best')

    f.text(0.0, 0.9, r'$\rm a)$', ha='center', fontsize=16)
    f.text(0.0, 0.48, r'$\rm b)$', ha='center', fontsize=16)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_2D_prob_single(target_dir):
    # Energy chapter. steady state probability
    output_file_name1 = (target_dir + "results/" + "Pss_2D_single_" +
                         "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")
    Ecouple = 16.0

    plt.figure()
    f1, ax1 = plt.subplots(1, 1, figsize=(3, 3))

    # plots
    if E0 == 0.0:
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    else:
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
        N = int(sqrt(len(data_array)))  # check grid size
        prob_ss_array = data_array.reshape((N, N))
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

    pmarg = trapz(prob_ss_array, axis=1)
    pcond = prob_ss_array / pmarg[:, None]

    cs = ax1.contourf(pcond.T, cmap=plt.cm.cool, vmin=0, vmax=amax(pcond))

    ax1.set_ylabel(r'$\theta_{\rm 1}$', fontsize=16)
    ax1.set_xlabel(r'$\theta_{\rm o}$', fontsize=16)
    ax1.set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
    ax1.set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
    ax1.set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])

    cax = f1.add_axes([0.94, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1
    )
    cbar.set_label(r'$p_{\rm ss}(\theta_1| \theta_{\rm o})$', fontsize=16)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    # f1.tight_layout()
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_2D_cm_rel_prob(target_dir):
    output_file_name1 = (target_dir + "results/" + "Pss_cm_diff.pdf")

    Ecouple_array = array([0.0, 4.0, 8.0, 16.0, 32.0, 128.0])
    Ecouplelabels = ['$0$', '$4$', '$8$', '$16$', '$32$', '$128$']

    plt.figure()
    f1, ax1 = plt.subplots(2, 3, figsize=(6, 4))

    # Find max prob. to set plot range
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, amax(Ecouple_array), E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
        N = int(sqrt(len(data_array)))
        prob_ss_array = data_array.reshape((N, N))
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, amax(Ecouple_array), E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

    prob_max = amax(prob_ss_array)

    # plots
    for ii, Ecouple in enumerate(Ecouple_array):
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0),
                usecols=0)
            N = int(sqrt(len(data_array)))  # check grid size
            prob_ss_array = data_array.reshape((N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        prob_new = zeros((N, N))
        for i in range(N):
            for j in range(N):
                # if (j < i and j + i < N) or (j > i and N < (j + i)):
                prob_new[i, (j + int(N/2)) % N] = prob_ss_array[(i + j) % N, (i - j) % N]

        cs = ax1[ii // 3, ii % 3].contourf(prob_new.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)

        if ii == 0:
            ax1[ii // 3, ii % 3].set_title(r"$\beta E_{\rm couple}$" + "={}".format(Ecouplelabels[ii]))
        else:
            ax1[ii // 3, ii % 3].set_title(Ecouplelabels[ii])
        if ii % 3 == 0:
            ax1[ii // 3, ii % 3].set_ylabel(r'$\Delta \theta/2$')
            ax1[ii // 3, ii % 3].set_yticklabels(['$-\pi$', '', '', '$0$', '', '', '$\pi$'])
        else:
            ax1[ii // 3, ii % 3].set_yticklabels(['', '', '', '', '', '', ''])
        if ii // 3 == 1:
            ax1[ii // 3, ii % 3].set_xlabel(r'$\bar{\theta}$')
            ax1[ii // 3, ii % 3].set_xticklabels(['$0$', '', '$\pi/3$', '', '$2 \pi /3$', '', '$2 \pi$'])
        else:
            ax1[ii // 3, ii % 3].set_xticklabels(['', '', '', '', '', '', ''])
        ax1[ii // 3, ii % 3].spines['right'].set_visible(False)
        ax1[ii // 3, ii % 3].spines['top'].set_visible(False)
        ax1[ii // 3, ii % 3].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[ii // 3, ii % 3].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])

    cax = f1.add_axes([0.94, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1, ticks=[0, 10**(-5), 2*10**(-5), 3*10**(-5), 4*10**(-5)]
    )
    cbar.set_label(r'$p_{\rm ss}(\bar{\theta}, \frac{\Delta \theta}{2})$', fontsize=14)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    plt.subplots_adjust(hspace=0.3)
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_entropy_production_Ecouple(target_dir):
    phase_shift = 0.0
    gamma = 1000
    barrier_height = [0.0, 2.0]

    output_file_name = (target_dir + "results/" +
                        "Entropy_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phase_{6}" + ".pdf")
    plt.figure()
    f, ax = plt.subplots(1, 2, figsize=(12, 4))

    for i, E0 in enumerate(barrier_height):
        E1 = E0
        # calculate entropy production
        if E0 == 0.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        else:
            Ecouple_array_tot = sort(
                concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))

        integrate_entropy_X = empty(Ecouple_array_tot.size)
        integrate_entropy_Y = empty(Ecouple_array_tot.size)
        integrate_entropy_sum = empty(Ecouple_array_tot.size)
        integrate_entropy_diff = empty(Ecouple_array_tot.size)

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if E0 == 0.0:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            else:
                if Ecouple in Ecouple_array_peak:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/201016_dip/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                            num_minima2, phase_shift),
                                     usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))
                dx = 2 * math.pi / N

                prob_ss_array = data_array[:, 0].reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

                for k in range(N):
                    for j in range(N):
                        if prob_ss_array[k, j] == 0:
                            prob_ss_array[k, j] = 10e-18

                flux_array = zeros((2, N, N))
                calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
                flux_array = asarray(flux_array)

                integrate_entropy_X[ii] = gamma * trapz(trapz(flux_array[0, ...]**2 / prob_ss_array)) * timescale
                integrate_entropy_Y[ii] = gamma * trapz(trapz(flux_array[1, ...]**2 / prob_ss_array)) * timescale

                integrate_entropy_sum[ii] = gamma * trapz(trapz(
                    (flux_array[0, ...] + flux_array[1, ...]) ** 2 / prob_ss_array)
                ) * timescale
                integrate_entropy_diff[ii] = gamma * trapz(trapz(
                    (flux_array[0, ...] - flux_array[1, ...]) ** 2 / prob_ss_array)
                ) * timescale

            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))

        # plot entropy production
        ax[i].plot(Ecouple_array_tot, integrate_entropy_X, '-o', label=r'$\dot{S}^{\rm o}_{\rm i}$', color='tab:blue')
        ax[i].plot(Ecouple_array_tot, integrate_entropy_Y, '-v', label=r'$\dot{S}^1_{\rm i}$', color='tab:blue')
        ax[i].plot(Ecouple_array_tot, integrate_entropy_Y + integrate_entropy_X, '-o', label=r'$\dot{S}_{\rm i}$',
                color='tab:orange')
        ax[i].plot(Ecouple_array_tot, 0.5*integrate_entropy_sum, '-o', label=r'$\dot{S}^{\bar{\theta}}_{\rm i}$',
                color='tab:green')
        ax[i].plot(Ecouple_array_tot, 0.5*integrate_entropy_diff, '-v', label=r'$\dot{S}^{\Delta \theta}_{\rm i}$',
                color='tab:green')

        ax[i].set_xlim((2, None))
        ax[i].set_ylim((3, 3*10**2))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
        ax[0].set_ylabel(r'$\dot{S}_{\rm i} \ \rm (s^{-1})$', fontsize=16)
        ax[i].tick_params(axis='both', labelsize=16)
        ax[i].yaxis.offsetText.set_fontsize(16)
        if i == 1:
            ax[i].legend(fontsize=16, frameon=False, ncol=2)
            ax[i].set_title(r'$2$', fontsize=16)
        else:
            ax[i].set_title(r'$\beta E_{\rm o} = \beta E_1 = 0$', fontsize=16)

    f.text(0.06, 0.9, r'$\rm a)$', fontsize=16)
    f.text(0.49, 0.9, r'$\rm b)$', fontsize=16)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift),
                  bbox_inches='tight')


def plot_power_entropy_correlation(target_dir):
    phase_shift = 0.0
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    gamma = 1000
    entropy_data = empty((psi1_array.size, psi_ratio.size, 3))
    power_data = empty((psi1_array.size, psi_ratio.size, 3))
    markerlst = ['s', 'D', 'o']
    colorlst = ['tab:blue', 'tab:orange', 'tab:green']

    output_file_name = (target_dir + "results/" + "Power_entropy_correlation" + ".pdf")

    # calculate entropy production rates and determine where curves cross
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print(psi_1, psi_2)

            if psi_1 == 4.0:
                Ecouple_array_tot = sort(
                    concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

            integrate_entropy_X = empty(Ecouple_array_tot.size)
            integrate_entropy_Y = empty(Ecouple_array_tot.size)

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if Ecouple in Ecouple_array_peak:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                elif psi_1 == 4.0:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/201016_dip/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                elif (psi_1 == 2.0 and psi_2 == -1.0 and Ecouple in Ecouple_array) or \
                        (psi_1 == 8.0 and Ecouple in Ecouple_array):
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                try:
                    data_array = loadtxt(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1,
                                                                num_minima2, phase_shift),
                                         usecols=(0, 2, 3, 4, 5, 6, 7, 8))
                    N = int(sqrt(len(data_array[:, 0])))
                    dx = 2 * math.pi / N
                    prob_ss_array = data_array[:, 0].reshape((N, N))
                    drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

                    for k in range(N):
                        for l in range(N):
                            if prob_ss_array[k, l] == 0:
                                prob_ss_array[k, l] = 10e-18

                    flux_array = zeros((2, N, N))
                    calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
                    flux_array = asarray(flux_array)

                    integrate_entropy_X[ii] = gamma * trapz(trapz(flux_array[0, ...]**2 / prob_ss_array)) * timescale
                    integrate_entropy_Y[ii] = gamma * trapz(trapz(flux_array[1, ...]**2 / prob_ss_array)) * timescale

                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                diff = abs(integrate_entropy_X[ii] - integrate_entropy_Y[ii])
                if abs(integrate_entropy_X[ii + 1] - integrate_entropy_Y[ii + 1]) > diff:
                    entropy_data[i, j, 0] = Ecouple_array_tot[ii]  # best estimate crossover
                    entropy_data[i, j, 1] = Ecouple_array_tot[ii] - Ecouple_array_tot[ii - 1]  # error bar size lower
                    entropy_data[i, j, 2] = Ecouple_array_tot[ii + 1] - Ecouple_array_tot[ii]  # error bar size upper
                    break

    # Figure out coupling strength that maximizes power
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                       "power_heat_info_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                       "_outfile.dat")

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            power_y_array = []
            if psi_1 == 4.0:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak,
                                                      Ecouple_array_quad)))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=2)

                    power_y = array(data_array)
                    power_y_array = append(power_y_array, power_y)
                except OSError:
                    print('Missing file power')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            print(Ecouple_array_tot.size, power_y_array.size)
            idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y_array.argmin()])).argmin()
            power_data[i, j, 0] = Ecouple_array_tot[idx]
            power_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            power_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

    plt.figure()
    f, ax = plt.subplots(1, 1)

    ax.plot(range(5, 25), range(5, 25), '--', color='gray')
    for i in range(3):
        for j in range(3):
            ax.errorbar(power_data[i, j, 0], entropy_data[i, j, 0], yerr=entropy_data[i, j, 1:3].T.reshape((2, 1)),
                        xerr=power_data[i, j, 1:3].T.reshape((2, 1)), marker=markerlst[j], fmt='', linestyle='None',
                        color=colorlst[i])

    ax.set_xlim((5, 25))
    ax.set_ylim((5, 25))
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$\beta E_{\rm couple} \, (\rm max \ power)$', fontsize=16)
    ax.set_ylabel(r'$\beta E_{\rm couple} \, (\rm entropy \ crossover)$', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.yaxis.offsetText.set_fontsize(16)

    f.savefig(output_file_name, bbox_inches='tight')


def plot_power_bound_Ecouple(target_dir):
    phase_array = array([0.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    barrier_height = array([2.0])
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Power_bound_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for k, phi in enumerate(phase_array):
                plt.figure()
                f, ax = plt.subplots(1, 1)
                for j, E0 in enumerate(barrier_height):
                    E1 = E0
                    if E0 == 0.0:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
                    else:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak,
                                                                Ecouple_array_quad)))

                    power_x = empty(Ecouple_array_total.size)
                    power_y = empty(Ecouple_array_total.size)
                    energy_xy = empty(Ecouple_array_total.size)
                    learning_rate = empty(Ecouple_array_total.size)

                    for i, Ecouple in enumerate(Ecouple_array_total):
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                            power_x[i] = data_array[1]
                            power_y[i] = data_array[2]
                            energy_xy[i] = data_array[5]
                            learning_rate[i] = data_array[6]
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                    ax.axhline(0, color='black')
                    # ax.axhline(1, color='grey')

                    ax.plot(Ecouple_array_total, power_x, '-o', label=r'$\beta \mathcal{P}_{\rm H^+}$', color='tab:blue')
                    ax.plot(Ecouple_array_total, -power_y, '-o', label=r'$-\beta \mathcal{P}_{\rm ATP}$', color='tab:orange')
                    ax.plot(Ecouple_array_total, -energy_xy - learning_rate, '-o',
                            label=r'$\beta \mathcal{P}_{\rm o \to 1} + \dot{I}_{\rm o}$', color='tab:gray')

                ax.set_ylim((7, 2 * 10 ** 2))
                ax.set_xlim((2, None))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylabel(r'$\beta \mathcal{P} \ \rm (s^{-1})$', fontsize=16)
                ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
                ax.legend(fontsize=16, frameon=False, ncol=1)
                ax.tick_params(axis='both', labelsize=14)

                f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_power_infoflow(target_dir):
    phase_array = array([0.0])
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    barrier_height = array([2.0])
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Power_infoflow_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            for k, phi in enumerate(phase_array):
                plt.figure()
                f, ax = plt.subplots(1, 1)
                for j, E0 in enumerate(barrier_height):
                    E1 = E0
                    if E0 == 0.0:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
                    elif E0 == 4.0:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double,
                                                                Ecouple_array_peak)))
                    else:
                        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak,
                                                                Ecouple_array_quad)))

                    power_x = empty(Ecouple_array_total.size)
                    power_y = empty(Ecouple_array_total.size)
                    energy_xy = empty(Ecouple_array_total.size)
                    learning_rate = empty(Ecouple_array_total.size)

                    for i, Ecouple in enumerate(Ecouple_array_total):
                        try:
                            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                            power_x[i] = data_array[1]
                            power_y[i] = data_array[2]
                            energy_xy[i] = data_array[5]
                            learning_rate[i] = data_array[6]
                        except OSError:
                            print('Missing file')
                            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

                    idx = (abs(Ecouple_array_total - Ecouple_array_total[power_y.argmin()])).argmin()
                    ax.fill_between([Ecouple_array_total[idx - 1], Ecouple_array_total[idx + 1]], -10 ** 2, 10 ** 2,
                                          facecolor='tab:orange', alpha=0.4)

                    idx = (abs(Ecouple_array_total - Ecouple_array_total[energy_xy.argmin()])).argmin()
                    ax.fill_between([Ecouple_array_total[idx - 1], Ecouple_array_total[idx + 1]], -10 ** 2, 10 ** 2,
                                          facecolor='tab:blue', alpha=0.4)

                    idx = (abs(Ecouple_array_total - Ecouple_array_total[learning_rate.argmax()])).argmin()
                    ax.fill_between([Ecouple_array_total[idx - 1], Ecouple_array_total[idx + 1]], -10 ** 2, 10 ** 2,
                                    facecolor='tab:green', alpha=0.4)

                    ax.plot(Ecouple_array_total, -energy_xy, '-o',
                            label=r'$\beta\mathcal{P}_{\rm o \to 1}$', color='tab:blue')
                    ax.plot(Ecouple_array_total, -power_y, '-o',
                            label=r'$-\beta\mathcal{P}_{\rm ATP}$', color='tab:orange')
                    ax.plot(Ecouple_array_total, learning_rate, '-o',
                            label=r'$\dot{I}_1$', color='tab:green')

                ax.set_ylim((0.2, 5 * 10 ** 1))
                ax.set_xlim((2, None))
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylabel(r'$\beta \mathcal{P} \ \rm (s^{-1})$', fontsize=16)
                ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
                ax.legend(fontsize=16, frameon=False, ncol=1, loc=(0.7, 0.3))
                ax.tick_params(axis='both', labelsize=14)

                f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_infoflow_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
    psi1_array = array([2.0, 4.0, 6.0, 8.0])
    psi_ratio = array([8, 4, 2, 1.5, 1.25, 1.125])
    phi = 0.0

    output_file_name = (
            target_dir + "results/" + "LearningRate_Ecouple_grid_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" +
            "_.pdf")

    f, axarr = plt.subplots(psi1_array.size, psi_ratio.size, sharex='all', sharey='all',
                            figsize=(3*psi1_array.size, psi_ratio.size))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            psi_2 = round(psi_2, 2)
            print(psi_2)

            if psi_1 == 6.0 or (psi_1 == 4.0 and psi_2 == -0.5):
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad)))
            elif psi_1 == 4.0 and (psi_2 == -1 or psi_2 == -0.5):
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad,
                                                      Ecouple_array_peak)))
            elif abs(psi_2) > psi_1/2:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, array([11.31, 22.63, 45.25, 90.51]))))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

            information = ones(Ecouple_array_tot.size)*(-10)
            output_power = zeros(Ecouple_array_tot.size)

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                                   "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                                   "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=(2, 6))
                    output_power[ii] = data_array[0]
                    information[ii] = data_array[1]
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            if (psi_1 != 6.0) or (psi_1 == 6.0 and psi_2 > -4.0):
                idx = (abs(Ecouple_array_tot - Ecouple_array_tot[output_power.argmin()])).argmin()
                axarr[i, j].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], 10 ** (-2), 10 ** 3,
                                         facecolor='tab:orange', alpha=0.4)

                for ii, power in enumerate(output_power):
                    if power > 0:
                        stall_coupling_idx = Ecouple_array_tot[ii]
                axarr[i, j].axvline(stall_coupling_idx, linestyle='dashed', color='gray')

            axarr[i, j].plot(Ecouple_array_tot, information, '-o', color='C1', markersize=6)

            axarr[i, j].yaxis.offsetText.set_fontsize(14)
            axarr[i, j].tick_params(axis='both', labelsize=14)
            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            axarr[i, j].set_ylim((0, 3.4))
            axarr[i, j].set_xlim((2, None))
            axarr[i, j].set_xticks((10, 100))

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi_ratio.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi1_array[i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.2f$' % psi_ratio[j], fontsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.02, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    f.text(0.05, 0.5, r'$\dot{I}_1 \ \rm (s^{-1})$', va='center', rotation='vertical', fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\beta \mu_{\rm H^+}\ (\rm rad^{-1})$', va='center', rotation=270, fontsize=24)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi))


def plot_2D_prob_double(target_dir):
    # Energy chapter. steady state probability
    output_file_name1 = ( target_dir + "results/" +
            "Pss_2D_single_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple = 16.0

    plt.figure()
    f1, ax1 = plt.subplots(1, 2, figsize=(4.5, 2))

    # max
    E0 = 2.0
    E1 = E0
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
        N = int(sqrt(len(data_array)))  # check grid size
        prob_ss_array = data_array.reshape((N, N))
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
    pmarg = trapz(prob_ss_array, axis=1)
    pcond = prob_ss_array / pmarg[:, None]
    max_prob = amax(pcond)

    for i, E0 in enumerate([0.0, 2.0]):
        E1 = E0
        # plots
        if E0 == 0.0:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/201112" +
                    "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        else:
            input_file_name = (
                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                    "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
            N = int(sqrt(len(data_array)))  # check grid size
            prob_ss_array = data_array.reshape((N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))

        pmarg = trapz(prob_ss_array, axis=1)
        pcond = prob_ss_array / pmarg[:, None]

        cs = ax1[i].contourf(pcond.T, cmap=plt.cm.cool, vmin=0, vmax=max_prob)

        ax1[i].set_xlabel(r'$\theta_{\rm o}$', fontsize=14)
        ax1[i].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
        ax1[i].set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])

        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)
        ax1[i].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[i].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])

    ax1[0].set_title(r'$\beta E_{\rm o} = \beta E_1 = %.f$' % 0)
    ax1[1].set_title(r'$%.f$' % 2)
    ax1[0].set_ylabel(r'$\theta_{\rm 1}$', fontsize=14)
    ax1[0].set_yticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
    ax1[1].set_yticklabels(['', '', '', '', '', '', ''])

    cax = f1.add_axes([0.94, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1
    )
    cbar.set_label(r'$p_{\rm ss}(\theta_1| \theta_{\rm o})$', fontsize=14)
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    f1.text(0.01, 0.9, r'$\rm a)$', fontsize=14)
    f1.text(0.49, 0.9, r'$\rm b)$', fontsize=14)

    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_super_grid_peak(target_dir):  # grid of plots of output power, entropy rate, learning rate
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))
    psi1_array = array([2.0, 4.0, 8.0])
    psi2_array = array([-1.0, -2.0, -4.0])
    phi = 0.0
    power_x = zeros((Ecouple_array_tot.size, psi1_array.size))
    power_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    heat_x = zeros((Ecouple_array_tot.size, psi1_array.size))
    heat_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    power_x_to_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    information_flow = zeros((Ecouple_array_tot.size, psi1_array.size))

    colorlst = ['C1', 'C9']
    labellst = ['$2$', '$4$']

    output_file_name = (
            target_dir + "results/" + "Super_grid_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_log_.pdf")

    f, axarr = plt.subplots(4, 3, sharex='all', sharey='row', figsize=(8, 8))

    # Barrier data
    for k, E0 in enumerate([2.0, 4.0]):
        E1 = E0
        for i, psi_1 in enumerate(psi1_array):
            psi_2 = psi2_array[i]
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "data/200915_energyflows/" + "E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                                   "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                                   "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                    power_x[ii, i] = data_array[1]
                    power_y[ii, i] = data_array[2]
                    heat_x[ii, i] = data_array[3]
                    heat_y[ii, i] = data_array[4]
                    power_x_to_y[ii, i] = data_array[5]
                    information_flow[ii, i] = data_array[6]
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            # plot line/bar at coupling strength corresponding to max power
            idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y[:, i].argmin()])).argmin()

            for j in range(4):
                axarr[j, i].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], -2, 10 ** 3,
                                         facecolor=colorlst[k], alpha=0.4)

            axarr[2, i].axhline(0, color='black')

            axarr[0, i].plot(Ecouple_array_tot, -power_y[:, i], 'o-', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[1, i].plot(Ecouple_array_tot, -power_x_to_y[:, i], 'o-', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[2, i].plot(Ecouple_array_tot, ((-heat_x[:, i] + information_flow[:, i]) - (-heat_y[:, i] - information_flow[:, i])),
                             'o-', color=colorlst[k], label=labellst[k], markersize=6)
            axarr[3, i].plot(Ecouple_array_tot, information_flow[:, i], 'o-', color=colorlst[k], label=labellst[k],
                             markersize=6)

            for j in range(4):
                axarr[j, i].yaxis.offsetText.set_fontsize(14)
                axarr[j, i].tick_params(axis='y', labelsize=14)
                axarr[j, i].tick_params(axis='x', labelsize=14)
                axarr[j, i].set_xscale('log')
                axarr[j, i].set_yscale('log')
                axarr[j, i].spines['right'].set_visible(False)
                axarr[j, i].spines['top'].set_visible(False)
                axarr[j, i].set_xlim((2, 150))

            axarr[0, i].set_ylim((0.05, 120))
            axarr[1, i].set_ylim((0.05, 200))
            axarr[2, i].set_yscale('linear')
            # axarr[2, i].set_ylim((10**(-2), 10**2))
            axarr[2, i].set_ylim((-2, 2))
            axarr[2, i].spines['bottom'].set_visible(False)
            axarr[3, i].set_ylim((0.1, 6.5))

            axarr[0, i].set_title(r'$%.0f$' % psi1_array[i], fontsize=18)

    axarr[0, 0].set_ylabel(r'$-\beta \mathcal{P}_{\rm ATP} \ \rm (s^{-1})$', fontsize=14)
    axarr[1, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm o \to 1} \ \rm (s^{-1})$', fontsize=14)
    axarr[2, 0].set_ylabel(r'$\dot{S}^{\rm o}_{\rm i} - \dot{S}^1_{\rm i} \ \rm (s^{-1})$', fontsize=14)
    axarr[3, 0].set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=14)
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$\beta \mu_{\rm H^+}\, (\rm rad^{-1})$', ha='center', fontsize=20)
    f.text(0.5, 0.05, r'$\beta E_{\rm couple}$', ha='center', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_info_Ecouple_N(target_dir):
    phi = 0.0
    dt = 5e-2
    output_file_name = (target_dir + "results/" + "Infoflow_Ecouple_finer_grid_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

    labels = [r'$360$', r'$720$']

    plt.figure()
    f, ax = plt.subplots(1, 1)
    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip/" +
                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                        "_outfile.dat")

    for j in range(2):
        learning_rate = empty(Ecouple_array_total.size)
        for i, Ecouple in enumerate(Ecouple_array_total):
            if j == 0:
                if Ecouple in Ecouple_array:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                elif Ecouple in Ecouple_array_double:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
                else:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
            else:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N  # spacing between gridpoints

                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            flux_array = zeros((2, N, N))
            calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

            dflux_array = empty((2, N, N))
            derivative_flux(flux_array, dflux_array, N, dx)

            learning_rate[i] = -trapz(trapz(dflux_array[1, ...] * log(prob_ss_array))) * timescale

        ax.plot(Ecouple_array_total, learning_rate, '-o', label=labels[j])

    ax.set_xlim((2, None))
    ax.set_ylim((0.1, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=16)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
    ax.legend(fontsize=14, frameon=False, ncol=1, title_fontsize=16, title='$N$')
    ax.tick_params(axis='both', labelsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_flux_1D(target_dir):
    # Energy chapter. energy flows vs coupling strength
    # input power, output power, heat flows X and Y, power from X to Y
    lines = ['solid', 'dashed', 'dashdot']
    labels = [r'$0$', r'$1/3$', r'$2/3$']
    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip/" +
                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                       "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Flux_1D_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1)

    Ecouple = 16.0
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, 0.0)
        )
        N = int(sqrt(len(data_array)))  # check grid size
        dx = 2 * math.pi / N  # spacing between gridpoints
        positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
        print('Grid size: ', N)

        prob_ss_array = data_array[:, 0].reshape((N, N))
        drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
        diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

    flux_array = zeros((2, N, N))
    calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

    ax.axhline(0, color='black')

    for i in range(3):
        ax.plot(positions, roll(flux_array[1, :, i*N//3], 2*i*N//3), label=labels[i], linestyle=lines[i])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(r'$\theta_{\rm o}$', fontsize=16)
    ax.set_ylabel(r'$J_1(\theta_{\rm o} - \theta_1, \theta_1)$', fontsize=16)
    ax.set_xticks([0, pi/3, 2*pi/3, pi, 4*pi/3, 5*pi/6, 2*pi])
    ax.set_yticks([0, 5*10**(-9), 10**(-8), 1.5*10**(-8) ])
    ax.set_xticklabels(['$0$', '', '$2 \pi/3$', '', '$4 \pi/3$', '', '$ 2\pi$'])
    ax.tick_params(axis='both', labelsize=14)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.legend(fontsize=14, frameon=False, loc='best', title=r'$\theta_1$', title_fontsize=16)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))


def plot_info_Ecouple_dt(target_dir):
    phi = 0.0
    output_file_name = (target_dir + "results/" + "Infoflow_Ecouple_dt_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

    labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$']
    times = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    plt.figure()
    f, ax = plt.subplots(1, 1)

    for j, dt in enumerate(times):
        learning_rate = empty(Ecouple_array_total.size)
        for i, Ecouple in enumerate(Ecouple_array_total):
            if Ecouple in Ecouple_array:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            elif Ecouple in Ecouple_array_double:
                input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                        "_outfile.dat")
            else:
                input_file_name = (
                        "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                        "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N  # spacing between gridpoints

                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            step_X = empty((N, N))
            step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

            mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
            Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

            pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
            Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

            learning_rate[i] = timescale * (Imem - Ipred) / dt

        ax.plot(Ecouple_array_total, learning_rate, '-o', label=labels[j])

    ax.set_xlim((2, None))
    ax.set_ylim((0.1, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=16)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
    ax.legend(fontsize=14, frameon=False, ncol=1, title_fontsize=16, title='$\Delta t$')
    ax.tick_params(axis='both', labelsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_info_Ecouple_steadystate(target_dir):
    phi = 0.0
    dt = 5e-2
    output_file_name = (target_dir + "results/" + "Infoflow_Ecouple_steadystate_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))

    # labels = [r'$360$', r'$720$']
    # labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$1$']
    labels = [r'$40\ \Delta t$', r'$80\ \Delta t$']
    # labels = [r'$\rm nostalgia$', r'$\partial_1 J$', r'$\rm conditional \ entropy$']
    # times = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    plt.figure()
    f, ax = plt.subplots(1, 1)
    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip/" +
                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                        "_outfile.dat")

    # for j, dt in enumerate(times):
    for j in range(2):
        learning_rate = empty(Ecouple_array_total.size)
        for i, Ecouple in enumerate(Ecouple_array_total):
            if j == 0:
                if Ecouple in Ecouple_array:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                elif Ecouple in Ecouple_array_double:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
                else:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
            else:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/201102_steadystate/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N  # spacing between gridpoints

                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            flux_array = zeros((2, N, N))
            calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

            dflux_array = empty((2, N, N))
            derivative_flux(flux_array, dflux_array, N, dx)

            learning_rate[i] = -trapz(trapz(dflux_array[1, ...] * log(prob_ss_array))) * timescale

        ax.plot(Ecouple_array_total, learning_rate, '-o', label=labels[j])

    ax.set_xlim((2, None))
    ax.set_ylim((0.1, 4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=16)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
    ax.legend(fontsize=14, frameon=False, ncol=1, title_fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_info_Ecouple_calcs(target_dir):
    phi = 0.0
    dt = 5e-2
    Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    labels = [r'$\rm nostalgia$', r'$\partial_1 J$', r'$\rm conditional \ entropy$']

    plt.figure()
    f, ax = plt.subplots(1, 1)

    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200921_dip/" +
                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                       "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Infoflow_Ecouple_calcs_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for j in range(3):
        learning_rate = empty(Ecouple_array_total.size)
        for i, Ecouple in enumerate(Ecouple_array_total):
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N  # spacing between gridpoints

                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 3:5].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 5:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            flux_array = zeros((2, N, N))
            calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

            if j == 0:
                step_X = empty((N, N))
                step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

                learning_rate[i] = timescale * (Imem - Ipred) / dt

            elif j == 1:
                dflux_array = empty((2, N, N))
                derivative_flux(flux_array, dflux_array, N, dx)

                learning_rate[i] = -trapz(trapz(dflux_array[1, ...] * log(prob_ss_array))) * timescale

            else:
                Hcouple = empty((N, N))
                for ii in range(N):
                    for jj in range(N):
                        if prob_ss_array[ii, jj] == 0:
                            prob_ss_array[ii, jj] = 10 ** (-18)

                        Hcouple[ii, jj] = -0.5 * Ecouple * sin(positions[ii] - positions[jj])

                dPxy = empty((N, N))
                calc_derivative(prob_ss_array, dPxy, N, dx, 1)

                Py = prob_ss_array.sum(axis=0)
                Pxgy = prob_ss_array / Py
                dPxgy = empty((N, N))
                calc_derivative(Pxgy, dPxgy, N, dx, 1)

                learning_rate[i] = -trapz(trapz(prob_ss_array * Hcouple * (dPxgy / Pxgy))) * timescale * 10 ** (-3) \
                                   - trapz(trapz(prob_ss_array * (dPxy / prob_ss_array) * (dPxgy / Pxgy))) * \
                                   timescale * 10 ** (-3)

        if E0 == 0:
            ax.plot(Ecouple_array_total, -learning_rate, 'o-', label=labels[j])
        else:
            ax.plot(Ecouple_array_total, learning_rate, 'o-', label=labels[j])

    ax.set_xlim((2, None))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=16)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
    ax.legend(fontsize=14, frameon=False, ncol=1, title_fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_info_Ecouple_error(target_dir):
    phi = 0.0
    barrier_height = [0.0, 2.0, 4.0]
    labels = [r'$0$', r'$2$', r'$4$']

    plt.figure()
    f, ax = plt.subplots(1, 1)

    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Infoflow_Ecouple_error_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    for j, E0 in enumerate(barrier_height):
        E1 = E0
        if E0 == 0:
            Ecouple_array_total = Ecouple_array
        else:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        learning_rate = empty(Ecouple_array_total.size)
        for i, Ecouple in enumerate(Ecouple_array_total):
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                learning_rate[i] = data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        if E0 == 0:
            ax.plot(Ecouple_array_total, -learning_rate, 'o-', label=labels[j])
        else:
            ax.plot(Ecouple_array_total, learning_rate, 'o-', label=labels[j])

    ax.set_xlim((2, None))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=16)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=16)
    ax.legend(fontsize=14, frameon=False, ncol=1, title_fontsize=16, title=r'$\beta E_{\rm o} = \beta E_1$')
    ax.tick_params(axis='both', labelsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_nn_learning_rate_Ecouple(input_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x', 'p']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    phi = 0.0
    learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(6, 4.5))
    axarr.axhline(0, color='black', label='_nolegend_')

    output_file_name = input_dir + "results/" + \
                       "Learning_rate_Ecouple_try3_scaled_nn_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log.pdf"

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                              "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                              "_outfile.dat"
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))
                learning_rate[ii, j] = data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

        axarr.plot((2*pi/num_min)*sqrt(Ecouple_array_tot), learning_rate[:, j],
                   color=color_lst[j], label=num_min, markersize=6, marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(16)
    axarr.tick_params(axis='both', labelsize=16)
    axarr.set_xlabel(r'$\frac{2 \pi}{n} \cdot \sqrt{ \beta E_{\rm couple} }$', fontsize=20)
    axarr.set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=20)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')
    axarr.set_yscale('log')
    axarr.set_ylim([2*10**(-2), None])

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o} = n_1$', fontsize=14,
                       loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(16)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi))


def plot_super_grid_nn(target_dir):  # grid of plots of output power, entropy rate, learning rate
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])
    phi = 0.0
    power_x = zeros((Ecouple_array_tot.size, min_array.size))
    power_y = zeros((Ecouple_array_tot.size, min_array.size))
    heat_x = zeros((Ecouple_array_tot.size, min_array.size))
    heat_y = zeros((Ecouple_array_tot.size, min_array.size))
    power_x_to_y = zeros((Ecouple_array_tot.size, min_array.size))
    information_flow = zeros((Ecouple_array_tot.size, min_array.size))

    output_file_name = (
            target_dir + "results/" + "Super_grid_nn_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_log_.pdf")

    f, axarr = plt.subplots(4, 5, sharex='all', sharey='row', figsize=(8, 7))

    # Barrier data
    for k, E0 in enumerate([2.0]):
        E1 = E0
        for i, num_min in enumerate(min_array):
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "data/200915_energyflows/" + "E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                                   "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                                   "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))
                    power_x[ii, i] = data_array[1]
                    power_y[ii, i] = data_array[2]
                    heat_x[ii, i] = data_array[3]
                    heat_y[ii, i] = data_array[4]
                    power_x_to_y[ii, i] = data_array[5]
                    information_flow[ii, i] = data_array[6]
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

            # plot line/bar at coupling strength corresponding to max power
            idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y[:, i].argmin()])).argmin()

            for j in range(4):
                axarr[j, i].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], -2, 10 ** 3,
                                         facecolor='tab:orange', alpha=0.4)

            axarr[2, i].axhline(0, color='black')

            axarr[0, i].plot(Ecouple_array_tot, -power_y[:, i], 'o-', color='tab:orange',
                             markersize=6)
            axarr[1, i].plot(Ecouple_array_tot, -power_x_to_y[:, i], 'o-', color='tab:orange',
                             markersize=6)
            axarr[2, i].plot(Ecouple_array_tot, ((-heat_x[:, i] + information_flow[:, i]) - (-heat_y[:, i] - information_flow[:, i])),
                             'o-', color='tab:orange', markersize=6)
            axarr[3, i].plot(Ecouple_array_tot, information_flow[:, i], 'o-', color='tab:orange',
                             markersize=6)

            for j in range(4):
                axarr[j, i].yaxis.offsetText.set_fontsize(14)
                axarr[j, i].tick_params(axis='y', labelsize=14)
                axarr[j, i].tick_params(axis='x', which='minor')
                axarr[j, i].tick_params(axis='x', labelsize=14)
                axarr[j, i].set_xscale('log')
                axarr[j, i].set_yscale('log')
                axarr[j, i].spines['right'].set_visible(False)
                axarr[j, i].spines['top'].set_visible(False)
                axarr[j, i].set_xlim((2, 150))
                axarr[j, i].set_xticks([10, 100])

            axarr[0, i].set_ylim((7, 25))
            axarr[1, i].set_ylim((5, 40))
            axarr[2, i].set_yscale('linear')
            # axarr[2, i].set_ylim((10**(-3), 2*10**2))
            axarr[2, i].set_ylim((-1.5, 10))
            axarr[2, i].spines['bottom'].set_visible(False)
            axarr[3, i].set_ylim((10**(-2), 10))

            axarr[0, i].set_title(r'$%.0f$' % min_array[i], fontsize=18)

    axarr[0, 0].set_ylabel(r'$-\beta \mathcal{P}_{\rm ATP} \ \rm (s^{-1})$', fontsize=14)
    axarr[1, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm o \to 1} \ \rm (s^{-1})$', fontsize=14)
    axarr[2, 0].set_ylabel(r'$\dot{S}^{\rm o}_{\rm i} - \dot{S}^1_{\rm i} \ \rm (s^{-1})$', fontsize=14)
    axarr[3, 0].set_ylabel(r'$\dot{I}_1 \ \rm (s^{-1})$', fontsize=14)
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$n_{\rm o} = n_1$', ha='center', fontsize=18)
    f.text(0.5, 0.05, r'$\beta E_{\rm couple}$', ha='center', fontsize=18)
    f.savefig(output_file_name.format(E0, E1, num_min, num_min, phi), bbox_inches='tight')


if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    heat_work_info(target_dir)
    # plot_power_efficiency_Ecouple_hor(target_dir)
    # plot_power_efficiency_Ecouple_ver(target_dir)
    # plot_2D_prob_flux_thesis()
    # plot_energy_flow(target_dir)
    # plot_power_Ecouple_grid(target_dir)
    # plot_efficiency_Ecouple_grid(target_dir)
    # plot_nn_power_efficiency_Ecouple(target_dir)
    # plot_n0_power_efficiency_Ecouple(target_dir)
    # plot_power_phi_single(target_dir)
    # plot_2D_prob_flux_phi_thesis(target_dir)
    # plot_nn_power_efficiency_phi(target_dir)
    # plot_energy_flow_zoom(target_dir)
    # plot_2D_prob_single(target_dir)
    # plot_2D_cm_rel_prob(target_dir)
    # plot_entropy_production_Ecouple(target_dir)
    # plot_power_entropy_correlation(target_dir)
    # plot_power_bound_Ecouple(target_dir)
    # plot_power_infoflow(target_dir)
    # plot_infoflow_Ecouple_grid(target_dir)
    # plot_2D_prob_double(target_dir)
    # plot_super_grid_peak(target_dir)
    # plot_info_Ecouple_N(target_dir)
    # plot_flux_1D(target_dir)
    # plot_info_Ecouple_dt(target_dir)
    # plot_info_Ecouple_steadystate(target_dir)
    # plot_info_Ecouple_calcs(target_dir)
    # plot_info_Ecouple_error(target_dir)
    # plot_nn_learning_rate_Ecouple(target_dir)
    # plot_super_grid_nn(target_dir)
