from numpy import array, linspace, loadtxt, pi, empty, sqrt, zeros, trapz, log, sin, append, amax, \
    concatenate, sort
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

N = 720  # N x N grid is used for Fokker-Planck simulations
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
    Ecouple_array_tot = array([64.0])
    psi1_array = array([8.0, 4.0])
    psi2_array = array([-4.0, -2.0, -1.0])
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
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/210521/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                    output_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                                        "power_heat_info_" +
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


def plot_energy_flow(target_dir):
    # Energy chapter. energy flows vs coupling strength
    # input power, output power, heat flows X and Y, power from X to Y
    phi = 0.0
    barrier_height = array([0.0, 2.0])
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Energy_flow_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(2, 1, figsize=(4, 6))
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

        ax[j].axhline(0, color='black')

        if j == 0:
            ax[0].plot(Ecouple_array_total, power_x, linestyle='dashed', marker='o', color='tab:blue')
            ax[0].plot(Ecouple_array_total, heat_x, linestyle='dashed', marker='o', color='tab:orange')
            ax[1].plot(Ecouple_array_total, -energy_xy, linestyle='dashed', marker='o', color='tab:green')
            ax[1].plot(Ecouple_array_total, heat_y, linestyle='dashed', marker='o', color='tab:red')
            ax[1].plot(Ecouple_array_total, power_y, linestyle='dashed', marker='o', color='tab:purple')
        else:
            ax[0].plot(Ecouple_array_total, power_x, linestyle='solid', marker='o', label=r'$\mathcal{P}_{\rm X}$',
                       color='tab:blue')
            ax[0].plot(Ecouple_array_total, heat_x, linestyle='solid', marker='o', label=r'$\dot{Q}_{\rm X}$',
                       color='tab:orange')
            ax[1].plot(Ecouple_array_total, -energy_xy, linestyle='solid', marker='o',
                       label=r'$\mathcal{P}_{\rm X \to Y}$', color='tab:green')
            ax[1].plot(Ecouple_array_total, heat_y, linestyle='solid', marker='o', label=r'$\dot{Q}_{\rm Y}$',
                       color='tab:red')
            ax[1].plot(Ecouple_array_total, power_y, linestyle='solid', marker='o', label=r'$\mathcal{P}_{\rm Y}$',
                       color='tab:purple')

        ax[j].set_xlim((2, None))
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['bottom'].set_visible(False)
        ax[j].set_xscale('log')
        ax[j].tick_params(axis='both', labelsize=12)
        ax[j].yaxis.offsetText.set_fontsize(12)

    ax[0].set_ylim((-250, 250))
    ax[1].set_ylim((-60, 60))
    ax[0].set_ylabel(r'$\textrm{Energy flow X} \ (k_{\rm B}T \cdot \rm s^{-1})$', fontsize=14)
    ax[1].set_ylabel(r'$\textrm{Energy flow Y} \ (k_{\rm B}T \cdot \rm s^{-1})$', fontsize=14)
    ax[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)

    f.legend(fontsize=14, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0.05, 1.1))
    f.text(0.0, 0.97, r'$\rm a)$', fontsize=14)
    f.text(0.0, 0.51, r'$\rm b)$', fontsize=14)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_entropy_production_Ecouple(target_dir):
    phase_shift = 0.0
    barrier_height = [0.0, 2.0]
    lines = ['dashed', 'solid']

    output_file_name = (target_dir + "results/" +
                        "Entropy_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phase_{6}" + ".pdf")
    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(5, 4))

    for i, E0 in enumerate(barrier_height):
        E1 = E0

        # calculate entropy production
        if E0 == 0.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        else:
            Ecouple_array_tot = sort(
                concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))

        heat_x = empty(Ecouple_array_tot.size)
        heat_y = empty(Ecouple_array_tot.size)
        learning_rate = empty(Ecouple_array_tot.size)

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                               "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                heat_x[ii] = data_array[3]
                heat_y[ii] = data_array[4]
                learning_rate[ii] = data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        # plot entropy production
        ax.plot(Ecouple_array_tot, -heat_x + learning_rate, linestyle=lines[i], marker='o',
                   label=r'$\dot{S}^{\rm X}_{\rm i}$', color='tab:orange')
        ax.plot(Ecouple_array_tot, -heat_y - learning_rate, linestyle=lines[i], marker='o',
                   label=r'$\dot{S}^{\rm Y}_{\rm i}$', color='tab:red')

    ax.set_xlim((2, None))
    ax.set_ylim((3, 3*10**2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax.set_ylabel(r'$\dot{S}_{\rm i} \, (\rm s^{-1})$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.offsetText.set_fontsize(12)
    # ax.legend(fontsize=14, frameon=False, ncol=1)
    ax.legend([r'$\dot{S}^{\rm X}_{\rm i}$', r'$\dot{S}^{\rm Y}_{\rm i}$', r'$\dot{S}^{\rm X}_{\rm i}$',
               r'$\dot{S}^{\rm Y}_{\rm i}$'], fontsize=14, frameon=False, ncol=2, loc=[0.45, 0.65])
    f.text(0.42, 0.8, r'$\beta E_{\rm X} = \beta E_{\rm Y} = 0$', fontsize=14)
    f.text(0.8, 0.805, r'$2$', fontsize=14)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), bbox_inches='tight')


def plot_power_bound_Ecouple(target_dir):
    phi = 0.0
    barrier_height = array([0.0, 2.0])
    lines = ['dashed', 'solid']
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Power_bound_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
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

        if j == 0:
            ax.plot(Ecouple_array_total, power_x, linestyle=lines[j], marker='o', color='tab:blue')
            ax.plot(Ecouple_array_total, -power_y, linestyle=lines[j], marker='o', color='tab:purple')
            ax.plot(Ecouple_array_total, -energy_xy - learning_rate, linestyle=lines[j], marker='o',
                    color='black')
        else:
            ax.plot(Ecouple_array_total, power_x, linestyle=lines[j], marker='o',
                    label=r'$\beta \mathcal{P}_{\rm X}$', color='tab:blue')
            ax.plot(Ecouple_array_total, -power_y, linestyle=lines[j], marker='o',
                    label=r'$-\beta \mathcal{P}_{\rm Y}$', color='tab:purple')
            ax.plot(Ecouple_array_total, -energy_xy - learning_rate, linestyle=lines[j], marker='o',
                    label=r'$\beta \mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X}$', color='black')

    ax.set_ylim((7, 3 * 10 ** 2))
    ax.set_xlim((2, None))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\beta \mathcal{P} \ (\rm s^{-1})$', fontsize=14)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax.legend(fontsize=14, frameon=False, ncol=1)
    ax.tick_params(axis='both', labelsize=12)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_nn_learning_rate_Ecouple(input_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x', 'p']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    phi = 0.0
    learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

    f, axarr = plt.subplots(2, 1, sharey='row', figsize=(5, 7))

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

        axarr[0].plot(Ecouple_array_tot, learning_rate[:, j],
                      color=color_lst[j], label=num_min, marker=markerlst[j], linestyle='-')
        axarr[1].plot((2*pi/num_min)*Ecouple_array_tot**0.5, learning_rate[:, j],
                   color=color_lst[j], label=num_min, marker=markerlst[j], linestyle='-')

    for i in range(2):
        # formatting
        axarr[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axarr[i].yaxis.offsetText.set_fontsize(12)
        axarr[i].tick_params(axis='both', labelsize=12)
        axarr[i].set_ylabel(r'$\dot{I}_1 \, (\rm s^{-1})$', fontsize=14)
        axarr[i].spines['right'].set_visible(False)
        axarr[i].spines['top'].set_visible(False)
        axarr[i].set_xscale('log')
        axarr[i].set_yscale('log')
        axarr[i].set_ylim([2*10**(-2), None])

    axarr[0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    axarr[1].set_xlabel(r'$\frac{2 \pi}{n} \cdot \sqrt{\beta E_{\rm couple} }$', fontsize=14)

    leg = axarr[1].legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o} = n_1$', fontsize=14, ncol=1,
                          frameon=False, bbox_to_anchor=(0.75, 0.25), title_fontsize=14)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.text(0.0, 0.87, r'$\rm a)$', fontsize=14)
    f.text(0.0, 0.45, r'$\rm b)$', fontsize=14)
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi), bbox_inches='tight')


def plot_nn_learning_rate_Ecouple_inset(input_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x', 'p']
    color_lst = ['C5', 'C6', 'C7', 'C8', 'C9', 'C9']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    phi = 0.0
    learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(6, 4))
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

        axarr.plot((2*pi/num_min)*Ecouple_array_tot**0.5, learning_rate[:, j],
                   color=color_lst[j], label=num_min, marker=markerlst[j], linestyle='-')

    # Inset, not scaled
    ax2 = inset_axes(axarr, width=2, height=1.4, loc='upper right')
    for j, num_min in enumerate(min_array):
        ax2.plot(Ecouple_array_tot, learning_rate[:, j], color=color_lst[j], marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(12)
    axarr.tick_params(axis='both', labelsize=12)
    axarr.set_xlabel(r'$\frac{2 \pi}{n} \cdot \sqrt{\beta E_{\rm couple} }$', fontsize=14)
    axarr.set_ylabel(r'$\dot{I}_{\rm Y} \, (\rm s^{-1})$', fontsize=14)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')
    axarr.set_yscale('log')
    axarr.set_ylim([2*10**(-2), None])
    axarr.set_xlim([2, 100])

    # inset formatting
    # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim([2*10**(-2), None])
    ax2.set_yticks([])
    ax2.set_xlabel(r'$\beta E_{\rm couple}$')

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm X} = n_{\rm Y}$', fontsize=14, ncol=3,
                       frameon=False, bbox_to_anchor=(0.85, 1.3), title_fontsize=14)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi), bbox_inches='tight')


def plot_power_entropy_correlation(target_dir):
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    entropy_data = empty((psi1_array.size, psi_ratio.size, 3))
    power_data = empty((psi1_array.size, psi_ratio.size, 3))
    power_xy_data = empty((psi1_array.size, psi_ratio.size, 3))
    infoflow_data = empty((psi1_array.size, psi_ratio.size, 3))
    markerlst = ['s', 'D', 'o']
    colorlst = ['darkblue', 'purple', 'darkred']

    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                       "power_heat_info_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                       "_outfile.dat")

    output_file_name = (target_dir + "results/" + "Power_entropy_correlation" + ".pdf")

    # calculate entropy production rates and determine where curves cross
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio
            print(psi_1, psi_2)
            power_y_array = []
            heat_x_array = []
            heat_y_array = []
            power_x_to_y_array = []
            info_array = []

            if psi_1 == 4.0:
                Ecouple_array_tot = sort(
                    concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

            if psi_1 == 4.0:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak,
                                                      Ecouple_array_quad)))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(2, 3, 4, 5, 6))

                    power_y = array(data_array[0])
                    heat_x = array(data_array[1])
                    heat_y = array(data_array[2])
                    power_x_to_y = array(data_array[3])
                    info_flow = array(data_array[4])
                    power_y_array = append(power_y_array, power_y)
                    heat_x_array = append(heat_x_array, heat_x)
                    heat_y_array = append(heat_y_array, heat_y)
                    power_x_to_y_array = append(power_x_to_y_array, power_x_to_y)
                    info_array = append(info_array, info_flow)
                except OSError:
                    print('Missing file power')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            # crossing entropy curves
            entropy_x_array = -heat_x_array + info_array
            entropy_y_array = -heat_y_array - info_array

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                diff = abs(entropy_x_array[ii] - entropy_y_array[ii])
                if abs(entropy_x_array[ii + 1] - entropy_y_array[ii + 1]) > diff:
                    entropy_data[i, j, 0] = Ecouple_array_tot[ii]  # best estimate crossover
                    entropy_data[i, j, 1] = Ecouple_array_tot[ii] - Ecouple_array_tot[ii - 1]  # error bar size lower
                    entropy_data[i, j, 2] = Ecouple_array_tot[ii + 1] - Ecouple_array_tot[ii]  # error bar size upper
                    break

            # max power
            #idx = (abs(Ecouple_array_tot - Ecouple_array_tot[power_y_array.argmin()])).argmin()
            idx = power_y_array.argmin()
            power_data[i, j, 0] = Ecouple_array_tot[idx]
            power_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            power_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

            # max transmitted power
            idx = power_x_to_y_array.argmin()
            power_xy_data[i, j, 0] = Ecouple_array_tot[idx]
            power_xy_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            power_xy_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

            # max information flow
            idx = info_array.argmax()
            infoflow_data[i, j, 0] = Ecouple_array_tot[idx]
            infoflow_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            infoflow_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

    plt.figure()
    f, ax = plt.subplots(3, 1, figsize=(4, 9))
    ax[0].plot(range(5, 25), range(5, 25), '-', color='gray')
    ax[1].plot(range(5, 25), range(5, 25), '-', color='gray')
    ax[2].plot(range(5, 25), range(5, 25), '-', color='gray')

    for i in range(3):
        for j in range(3):
            ax[0].errorbar(power_data[i, j, 0], entropy_data[i, j, 0], yerr=entropy_data[i, j, 1:3].T.reshape((2, 1)),
                           xerr=power_data[i, j, 1:3].T.reshape((2, 1)), marker=markerlst[j], fmt='', linestyle='None',
                           color=colorlst[i])
            ax[1].errorbar(power_data[i, j, 0], power_xy_data[i, j, 0], yerr=power_xy_data[i, j, 1:3].T.reshape((2, 1)),
                           xerr=power_data[i, j, 1:3].T.reshape((2, 1)), marker=markerlst[j], fmt='', linestyle='None',
                           color=colorlst[i])
            ax[2].errorbar(power_data[i, j, 0], infoflow_data[i, j, 0], yerr=infoflow_data[i, j, 1:3].T.reshape((2, 1)),
                           xerr=power_data[i, j, 1:3].T.reshape((2, 1)), marker=markerlst[j], fmt='', linestyle='None',
                           color=colorlst[i])

        ax[i].set_xlim((5, 25))
        ax[i].set_ylim((5, 25))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].yaxis.offsetText.set_fontsize(12)
        ax[i].set_yticks([5, 10, 15, 20, 25])
        ax[i].set_xticklabels(["", "", "", "", ""])

    ax[2].set_xticks([5, 10, 15, 20, 25])
    ax[2].set_xticklabels(['$5$', '$10$', '$15$', '$20$', '$25$'])
    ax[2].set_xlabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ \mathcal{P}_{\rm Y}$', fontsize=14)
    ax[0].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ (\dot{S}^{\rm Y}_{\rm i} - \dot{S}^{\rm X}_{\rm i})$', fontsize=14)
    ax[1].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ \mathcal{P}_{\rm X \to Y}$', fontsize=14)
    ax[2].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ \dot{I}_{\rm Y}$', fontsize=14)

    f.text(-0.03, 0.88, r'$\rm a)$', fontsize=14)
    f.text(-0.03, 0.61, r'$\rm b)$', fontsize=14)
    f.text(-0.03, 0.34, r'$\rm c)$', fontsize=14)
    f.savefig(output_file_name, bbox_inches='tight')


def plot_2D_prob_triple(target_dir):
    # Energy chapter. steady state probability
    output_file_name1 = ( target_dir + "results/" +
            "Pss_2D_single_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouplelst = [0.0, 16.0, 128.0]

    plt.figure()
    f1, ax1 = plt.subplots(1, 3, figsize=(4, 1.2))

    # max
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouplelst[-1], E1, psi_1, psi_2, num_minima1, num_minima2, 0.0), usecols=0)
        N = int(sqrt(len(data_array)))  # check grid size
        prob_ss_array = data_array.reshape((N, N))
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, Ecouplelst[-1], E1, psi_1, psi_2, num_minima1, num_minima2, 0.0))
    pmarg = trapz(prob_ss_array, axis=1)
    pcond = prob_ss_array / pmarg[:, None]
    max_prob = amax(pcond)

    for i, Ecouple in enumerate(Ecouplelst):
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

        ax1[i].set_xlabel(r'$x \ (\rm rev)$', fontsize=10)
        ax1[i].set_xticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])

        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)
        ax1[i].set_xticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[i].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[i].set_title(r'$%.f$' % Ecouplelst[i], fontsize=10)
        ax1[i].set_yticklabels(['', '', '', '', '', '', ''])
        ax1[i].tick_params(axis='both', labelsize=8)
        ax1[i].yaxis.offsetText.set_fontsize(8)

    ax1[0].set_title(r'$\beta E_{\rm couple} = %.f$' % Ecouplelst[0], fontsize=10)
    ax1[0].set_ylabel(r'$y \ (\rm rev)$', fontsize=10)
    ax1[0].set_yticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])
    ax1[1].set_yticklabels(['', '', '', '', '', '', ''])

    f1.subplots_adjust(wspace=0.1)

    cax = f1.add_axes([0.16, -0.37, 0.7, 0.07])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='horizontal', ax=ax1,
    )
    cbar.set_label(r'$p_{\rm ss}(y|x)$', fontsize=10)
    cbar.formatter.set_scientific(True)
    cbar.ax.tick_params(labelsize=8)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    f1.text(0.05, 0.95, r'$\rm a)$', fontsize=10)
    f1.text(0.35, 0.95, r'$\rm b)$', fontsize=10)
    f1.text(0.62, 0.95, r'$\rm c)$', fontsize=10)

    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    # plot_energy_flow(target_dir)
    # plot_entropy_production_Ecouple(target_dir)
    # plot_power_bound_Ecouple(target_dir)
    # plot_nn_learning_rate_Ecouple(target_dir)
    # plot_nn_learning_rate_Ecouple_inset(target_dir)
    # plot_power_entropy_correlation(target_dir)
    plot_2D_prob_triple(target_dir)
