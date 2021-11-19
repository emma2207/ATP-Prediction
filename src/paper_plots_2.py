from numpy import array, linspace, loadtxt, pi, empty, sqrt, zeros, trapz, log, sin, append, amax, \
    concatenate, sort
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
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
Ecouple_array_double2 = array([2.83, 5.66, 11.31, 22.62, 45.25, 90.51])
Ecouple_array_quad = array([2.38, 3.36, 4.76, 6.73, 9.51, 13.45, 19.03, 26.91, 38.05, 53.82, 76.11, 107.63])

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
                    # needs a minus sign to make it \mathcal{P}_{\rm X \to Y}

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
                    # needs a minus sign to make it \dot{I}_{\rm X}

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
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_log_.pdf")
    ls = ['dashed', 'solid']

    plt.figure()
    f, ax = plt.subplots(2, 1, figsize=(4, 6.5))
    for j, E0 in enumerate(barrier_height):
        E1 = E0
        if E0 == 0.0:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad, Ecouple_array_peak)))
        else:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))

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

        ax[0].plot(Ecouple_array_total, power_x, linestyle=ls[j], marker='o', color='tab:blue', markersize=5)
        ax[0].plot(Ecouple_array_total, -heat_x, linestyle=ls[j], marker='o', color='tab:orange', markersize=5)
        ax[1].plot(Ecouple_array_total, -heat_y, linestyle=ls[j], marker='o', color='tab:red', markersize=5)
        ax[1].plot(Ecouple_array_total, -energy_xy, linestyle=ls[j], marker='o', color='tab:green', markersize=5)
        ax[1].plot(Ecouple_array_total, -power_y, linestyle=ls[j], marker='o', color='tab:purple', markersize=5)

        ax[j].set_xlim((2, None))
        ax[j].spines['right'].set_visible(False)
        ax[j].spines['top'].set_visible(False)
        ax[j].spines['bottom'].set_visible(False)
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].tick_params(axis='both', labelsize=12)
        ax[j].yaxis.offsetText.set_fontsize(12)

    ax[0].set_ylim((None, 250))
    ax[1].set_ylim((4, 60))
    ax[0].set_ylabel(r'$(\rm s^{-1})$', fontsize=14)
    ax[1].set_ylabel(r'$(\rm s^{-1})$', fontsize=14)
    ax[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)

    f.legend(handles=[Line2D([0], [0], color='black', linestyle='dashed', lw=2, label=r'$\beta E^{\ddagger} = 0$'),
                      Line2D([0], [0], color='black', linestyle='solid', lw=2, label=r'$\beta E^{\ddagger} = 2$')],
             loc=[0.2, 0.48], frameon=False, fontsize=14, ncol=2)

    f.text(0.7, 0.73, r'$\beta \mathcal{P}_{\rm X}$', fontsize=14, color='tab:blue')
    f.text(0.3, 0.66, r'$-\beta \dot{Q}_{\rm X}$', fontsize=14, color='tab:orange')
    f.text(0.6, 0.31, r'$-\beta \mathcal{P}_{\rm Y}$', fontsize=14, color='tab:purple')
    f.text(0.14, 0.33, r'$-\beta \dot{Q}_{\rm Y}$', fontsize=14, color='tab:red')
    f.text(0.75, 0.365, r'$\beta \mathcal{P}_{\rm X \to Y}$', fontsize=14, color='tab:green')
    f.text(0.0, 0.88, r'$\rm a)$', fontsize=14)
    f.text(0.0, 0.45, r'$\rm b)$', fontsize=14)

    f.subplots_adjust(hspace=0.4)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_entropy_production_Ecouple(target_dir):
    phase_shift = 0.0
    barrier_height = [2.0]
    lines = ['solid']
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
        ax.plot(Ecouple_array_tot, -heat_x + learning_rate, linestyle=lines[i], marker='o', color='tab:green')
        ax.plot(Ecouple_array_tot, -heat_y - learning_rate, linestyle=lines[i], marker='o', color='tab:red')

    ax.set_xlim((2, None))
    ax.set_ylim((3, 2.5*10**2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax.set_ylabel(r'$\dot{\Sigma} \, (\rm s^{-1})$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.yaxis.offsetText.set_fontsize(12)

    # f.legend(handles=[Line2D([0], [0], color='black', linestyle='dashed', lw=2, label=r'$0$'),
    #                   Line2D([0], [0], color='black', linestyle='solid', lw=2, label=r'$2$')],
    #          loc=[0.75, 0.7], frameon=False, fontsize=14, ncol=1, title=r'$\beta E^{\ddagger}$', title_fontsize=14)
    f.text(0.35, 0.75, r'$\dot{\Sigma}^{\rm o}$', fontsize=14, color='tab:green')
    f.text(0.2, 0.45, r'$\dot{\Sigma}^{\rm 1}$', fontsize=14, color='tab:red')

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
        Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double2, Ecouple_array_peak,
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
            ax.plot(Ecouple_array_total, power_x, linestyle=lines[j], marker='o', color='tab:blue')
            ax.plot(Ecouple_array_total, -power_y, linestyle=lines[j], marker='o', color='tab:purple')
            ax.plot(Ecouple_array_total, -energy_xy - learning_rate, linestyle=lines[j], marker='o', color='black')

    ax.set_ylim((7, 3 * 10 ** 2))
    ax.set_xlim((2, None))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\beta \mathcal{P} \ (\rm s^{-1})$', fontsize=14)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    f.text(0.2, 0.7, r'$\mathcal{P}_{\rm X}$', fontsize=14, color='tab:blue')
    f.text(0.14, 0.35, r'$\mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X}$', fontsize=14, color='black')
    f.text(0.44, 0.15, r'$-\mathcal{P}_{\rm Y}$', fontsize=14, color='tab:purple')

    f.legend(handles=[Line2D([0], [0], color='gray', linestyle='dashed', lw=2, label=r'$0$'),
                      Line2D([0], [0], color='gray', linestyle='solid', lw=2, label=r'$2$')],
             loc=[0.77, 0.72], frameon=False, fontsize=14, ncol=1, title=r'$\beta E^{\ddagger}$', title_fontsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_nn_learning_rate_Ecouple(input_dir):  # plot power and efficiency as a function of the coupling strength
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad, Ecouple_array_peak)))
    phi = 0.0
    learning_rate = zeros(Ecouple_array_tot.size)

    f, axarr = plt.subplots(1, 1, sharey='row', figsize=(6, 4))

    output_file_name = input_dir + "results/" + \
                       "Learning_rate_Ecouple_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log.pdf"

    # Fokker-Planck results (barriers)
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        input_file_name = target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                          "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                          "_outfile.dat"
        try:
            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
            learning_rate[ii] = data_array[6]
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

    axarr.plot(Ecouple_array_tot, learning_rate, color='C6', linestyle='-', marker='o')

    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(12)
    axarr.tick_params(axis='both', labelsize=12)
    axarr.set_ylabel(r'$-\dot{I}_{\rm o} \, (\rm s^{-1})$', fontsize=14)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')
    axarr.set_yscale('log')
    axarr.set_ylim([7*10**(-2), None])

    axarr.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi), bbox_inches='tight')


def plot_nn_learning_rate_Ecouple_inset(input_dir):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x', 'p']
    color_lst = ['C5', 'C6', 'C7', 'C8', 'C9', 'C9']
    phi = 0.0

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(6, 4))
    axarr.axhline(0, color='black', label='_nolegend_')
    ax2 = inset_axes(axarr, width=2, height=1.4, loc='upper right')

    output_file_name = input_dir + "results/" + \
                       "Learning_rate_Ecouple_try3_scaled_nn_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log.pdf"

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):

        Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        if num_min == 1.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array_tot, [0.16, 0.22, 0.31, 0.44, 0.63, 0.89, 1.26, 1.78,
                                                                      2.51, 3.56, 5.03, 7.11, 10.06, 14.22])))
        elif num_min == 2.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array_tot, [0.63, 0.89, 1.26, 1.78, 2.52, 3.56, 5.03, 7.11,
                                                                      10.06, 14.22, 20.11, 28.44, 40.23, 56.89])))
        elif num_min == 6.0 or num_min == 12.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array_tot, [181.0, 256.0, 362.0, 512.0])))
        # elif num_min == 12.0:
        #     Ecouple_array_tot = sort(concatenate((Ecouple_array_tot, [181.0, 256.0, 362.0, 512.0])))

        learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

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
        ax2.plot(Ecouple_array_tot, learning_rate[:, j], color=color_lst[j], marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(12)
    axarr.tick_params(axis='both', labelsize=12)
    axarr.set_xlabel(r'$\frac{2 \pi}{n} \cdot \sqrt{\beta E_{\rm couple} }$', fontsize=14)
    axarr.set_ylabel(r'$-\dot{I}_{\rm X} \, (\rm s^{-1})$', fontsize=14)
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

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n$', fontsize=14, ncol=3,
                       frameon=False, bbox_to_anchor=(0.85, 1.3), title_fontsize=14)
    leg_title = leg.get_title()
    leg_title.set_fontsize(14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi), bbox_inches='tight')


def plot_power_entropy_correlation(target_dir):
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    entropy_data = empty((psi1_array.size, psi_ratio.size, 3))
    power_data = empty((psi1_array.size, psi_ratio.size, 3))
    bound_data = empty((psi1_array.size, psi_ratio.size, 3))
    markerlst = ['o', 's', 'D']
    sizes = [300, 100, 100]
    barwidths = [4, 2, 2]
    colorlst = ['darkblue', 'firebrick', 'purple']

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
            power_x_array = []

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
                        usecols=(1, 2, 3, 4, 5, 6))

                    power_x = array(data_array[0])
                    power_y = array(data_array[1])
                    heat_x = array(data_array[2])
                    heat_y = array(data_array[3])
                    power_x_to_y = array(data_array[4])
                    info_flow = array(data_array[5])

                    power_x_array = append(power_x_array, power_x)
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

            ratio = -power_x_array * power_y_array / (power_x_to_y_array + info_array) ** 2

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                diff = abs(entropy_x_array[ii] - entropy_y_array[ii])
                if abs(entropy_x_array[ii + 1] - entropy_y_array[ii + 1]) > diff:
                    entropy_data[i, j, 0] = Ecouple_array_tot[ii]  # best estimate crossover
                    entropy_data[i, j, 1] = Ecouple_array_tot[ii] - Ecouple_array_tot[ii - 1]  # error bar size lower
                    entropy_data[i, j, 2] = Ecouple_array_tot[ii + 1] - Ecouple_array_tot[ii]  # error bar size upper
                    break

            # max power
            idx = power_y_array.argmin()
            power_data[i, j, 0] = Ecouple_array_tot[idx]
            power_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            power_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

            # max transmitted power
            # idx = power_x_to_y_array.argmin()
            # power_xy_data[i, j, 0] = Ecouple_array_tot[idx]
            # power_xy_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            # power_xy_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]
            #
            # # max information flow
            # idx = info_array.argmax()
            # infoflow_data[i, j, 0] = Ecouple_array_tot[idx]
            # infoflow_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            # infoflow_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

            # max bound
            idx = (-power_x_to_y_array - info_array).argmax()
            bound_data[i, j, 0] = Ecouple_array_tot[idx]
            bound_data[i, j, 1] = Ecouple_array_tot[idx] - Ecouple_array_tot[idx - 1]
            bound_data[i, j, 2] = Ecouple_array_tot[idx + 1] - Ecouple_array_tot[idx]

            # if i == 1 and j == 2:
            #     print(Ecouple_array_tot[idx])
            #     print(-power_x_to_y_array)
            #     print(info_array)

            dratio = zeros(Ecouple_array_tot.size - 1)

            for ii in range(len(dratio)):
                dratio[ii] = (ratio[ii + 1] - ratio[ii]) / (Ecouple_array_tot[ii + 1] - Ecouple_array_tot[ii])

            for ii in range(len(dratio) - 1, 0, -1):
                if dratio[ii] < 0:
                    print(Ecouple_array_tot[ii])
                    break

    # print(bound_data)

    plt.figure()
    f, ax = plt.subplots(2, 1, figsize=(4, 8.5))
    ax[0].plot(range(5, 25), range(5, 25), '--', color='gray', zorder=0)
    ax[1].plot(range(5, 25), range(5, 25), '--', color='gray', zorder=0)
    # ax[2].plot(range(5, 25), range(5, 25), '--', color='gray')

    for i in range(3):
        for j in range(3):
            markers, caps, bars = ax[1].errorbar(power_data[i, j, 0], bound_data[i, j, 0], yerr=bound_data[i, j, 1:3].T.reshape((2, 1)),
                           xerr=power_data[i, j, 1:3].T.reshape((2, 1)), fmt='', color=colorlst[j], marker=None,
                                                 elinewidth=barwidths[j], alpha=1, zorder=1)
            ax[1].scatter(power_data[i, j, 0], bound_data[i, j, 0], marker=markerlst[i], linestyle='None',
                           color=colorlst[j], s=sizes[j], alpha=1)
            # [bar.set_alpha(0.5) for bar in bars]
            markers, caps, bars = ax[0].errorbar(power_data[i, j, 0], entropy_data[i, j, 0], yerr=entropy_data[i, j, 1:3].T.reshape((2, 1)),
                           xerr=entropy_data[i, j, 1:3].T.reshape((2, 1)), fmt='', color=colorlst[j], marker=None,
                                                 elinewidth=barwidths[j], alpha=1, zorder=1)
            ax[0].scatter(power_data[i, j, 0], entropy_data[i, j, 0], marker=markerlst[i], linestyle='None',
                          color=colorlst[j], s=sizes[j], alpha=1)
            # [bar.set_alpha(0.5) for bar in bars]
            # markers, caps, bars = ax[2].errorbar(power_data[i, j, 0], infoflow_data[i, j, 0], yerr=infoflow_data[i, j, 1:3].T.reshape((2, 1)),
            #                xerr=power_data[i, j, 1:3].T.reshape((2, 1)), marker=markerlst[j], fmt='', linestyle='None',
            #                color=colorlst[i])
            # [bar.set_alpha(0.5) for bar in bars]

    for i in range(2):
        ax[i].set_xlim((5, 25))
        ax[i].set_ylim((5, 25))
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].yaxis.offsetText.set_fontsize(12)
        ax[i].set_yticks([5, 10, 15, 20, 25])
        ax[i].set_xticks([5, 10, 15, 20, 25])
        ax[i].set_xticklabels(["", "", "", "", ""])

    ax[-1].set_xticks([5, 10, 15, 20, 25])
    ax[-1].set_xticklabels(['$5$', '$10$', '$15$', '$20$', '$25$'])
    ax[-1].set_xlabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ \beta \mathcal{P}_{\rm Y}$', fontsize=14)
    ax[1].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmin}} \ |\dot{\Sigma}_{\rm Y} - \dot{\Sigma}_{\rm X}|$', fontsize=14)
    ax[0].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ (\beta \mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X})$', fontsize=14)
    # ax[2].set_ylabel(r'$\underset{\beta E_{\rm couple}}{\textrm{argmax}} \ \dot{I}_{\rm Y}$', fontsize=14)

    f.legend(handles=[Line2D([0], [0], color=colorlst[0], lw=4, label=r'$2$', alpha=1),
                      Line2D([0], [0], color=colorlst[1], lw=2, label=r'$4$', alpha=1),
                      Line2D([0], [0], color=colorlst[2], lw=2, label=r'$8$', alpha=1)],
             loc=[0.75, 0.58], frameon=False, fontsize=14, ncol=1, title=r'$-\mu_{\rm X}/\mu_{\rm Y}$', title_fontsize=14)

    f.legend(handles=[Line2D([0], [0], marker=markerlst[0], color='black', lw=0, label=r'$2$', alpha=1),
                      Line2D([0], [0], marker=markerlst[1], color='black', lw=0, label=r'$4$', alpha=1),
                      Line2D([0], [0], marker=markerlst[2], color='black', lw=0, label=r'$8$', alpha=1)],
             loc=[0.75, 0.1], frameon=False, fontsize=14, ncol=1, title=r'$\beta \mu_{\rm X}$', title_fontsize=14)

    f.subplots_adjust(hspace=0.1)
    f.text(-0.03, 0.88, r'$\rm a)$', fontsize=14)
    f.text(-0.03, 0.48, r'$\rm b)$', fontsize=14)
    # f.text(-0.03, 0.34, r'$\rm c)$', fontsize=14)
    f.savefig(output_file_name, bbox_inches='tight')


def plot_2D_prob_triple(target_dir):
    # Energy chapter. steady state probability
    output_file_name1 = ( target_dir + "results/" +
            "Pss_2D_single_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouplelst = [0.0, 16.0, 128.0]

    plt.figure()
    f1, ax1 = plt.subplots(2, 3, figsize=(4, 3), sharey='row')

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

        # heatmap part
        cs = ax1[0, i].contourf((N/(2*pi))*pcond.T, cmap=plt.cm.cool, vmin=0, vmax=max_prob*(N/(2*pi)))

        ax1[0, i].axvline(5 * N / 12, color='darkorange')
        ax1[0, i].axvline(3 * N / 12, color='orangered')

        ax1[0, i].spines['right'].set_visible(False)
        ax1[0, i].spines['top'].set_visible(False)
        ax1[0, i].spines['left'].set_visible(False)
        ax1[0, i].set_yticks([0, N/6, N/3, N/2, 2*N/3, 5*N/6, N])
        ax1[0, i].set_title(r'$%.f$' % Ecouplelst[i], fontsize=10)
        ax1[0, i].tick_params(axis='both', labelsize=8)
        ax1[0, i].yaxis.offsetText.set_fontsize(8)
        ax1[0, i].set_xlabel(r'$\theta_{\rm o}$', fontsize=10)
        ax1[0, i].set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        ax1[0, i].set_xticklabels(['$0$', '', r'$\frac{\pi}{3}$', '', r'$\frac{2 \pi}{3}$', '', r'$2 \pi$'])

        # slices
        ax1[1, i].plot((N/(2*pi))*pcond[3 * N // 12], color='orangered')
        ax1[1, i].plot((N/(2*pi))*pcond[5 * N // 12], color='darkorange')

        ax1[1, i].spines['right'].set_visible(False)
        ax1[1, i].spines['top'].set_visible(False)
        ax1[1, i].spines['left'].set_visible(False)
        ax1[1, i].set_xlabel(r'$\theta_1$', fontsize=10)
        ax1[1, i].set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        ax1[1, i].set_xticklabels(['$0$', '', r'$\frac{\pi}{3}$', '', r'$\frac{2 \pi}{3}$', '', r'$2 \pi$'])
        # ax1[1, i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax1[1, i].tick_params(axis='both', labelsize=8)
        ax1[1, i].yaxis.offsetText.set_fontsize(8)
        ax1[1, i].set_xlim((0, N))
        ax1[1, i].set_ylim((0, 3.5))
        ax1[1, i].set_yticks([0, 1, 2, 3])

    ax1[0, 0].set_title(r'$\beta E_{\rm couple} = %.f$' % Ecouplelst[0], fontsize=10)
    ax1[0, 0].set_ylabel(r'$\theta_1$', fontsize=10)
    ax1[0, 0].set_yticklabels(['$0$', '', r'$\frac{\pi}{3}$', '', r'$\frac{2 \pi}{3}$', '', r'$2 \pi$'])
    ax1[1, 0].set_ylabel(r'$p_{\rm ss}(\theta_1|\theta_{\rm o})$', fontsize=10)

    f1.subplots_adjust(hspace=0.65)

    cax = f1.add_axes([0.16, 1.1, 0.7, 0.04])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='horizontal', ax=ax1,
    )
    cbar.formatter.set_scientific(True)
    cbar.ax.tick_params(labelsize=8)
    # cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    f1.text(0.45, 1.2, r'$p_{\rm ss}(\theta_1|\theta_{\rm o})$', fontsize=10)
    f1.text(0.05, 0.9, r'$\rm a)$', fontsize=10)
    f1.text(0.35, 0.9, r'$\rm b)$', fontsize=10)
    f1.text(0.62, 0.9, r'$\rm c)$', fontsize=10)
    f1.text(0.05, 0.42, r'$\rm d)$', fontsize=10)
    f1.text(0.35, 0.42, r'$\rm e)$', fontsize=10)
    f1.text(0.62, 0.42, r'$\rm f)$', fontsize=10)

    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_lr_prob_slice(target_dir):
    # Energy chapter. steady state probability
    output_file_name1 = ( target_dir + "results/" +
            "LR_Pss_2D_slice_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}" + "_.pdf")

    Ecouplelst = [2.0, 16.0, 128.0]

    f1 = plt.figure(constrained_layout=True, figsize=(4, 5.5))

    widths = [1, 1, 1]
    heights = [1.5, 0.15, 1, 1]
    c1 = 'darkorange'
    c2 = 'orangered'

    gs = f1.add_gridspec(nrows=4, ncols=3, width_ratios=widths, height_ratios=heights)
    ax1 = f1.add_subplot(gs[0, :])
    axax = f1.add_subplot(gs[1, :])
    ax2 = f1.add_subplot(gs[2, 0])
    ax3 = f1.add_subplot(gs[2, 1])
    ax4 = f1.add_subplot(gs[2, 2])
    ax5 = f1.add_subplot(gs[3, 0])
    ax6 = f1.add_subplot(gs[3, 1])
    ax7 = f1.add_subplot(gs[3, 2])

    # probability plots
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

        # heatmap part
        if i == 0:
            ax2.contourf(N * pcond.T, cmap=plt.cm.cool, vmin=0, vmax=max_prob * N)
        elif i == 1:
            ax3.contourf(N * pcond.T, cmap=plt.cm.cool, vmin=0, vmax=max_prob * N)
        elif i == 2:
            cs = ax4.contourf(N * pcond.T, cmap=plt.cm.cool, vmin=0, vmax=max_prob * N)

        # slices
        if i == 0:
            ax5.plot(N*pcond[3 * N // 12], color=c2)
            ax5.plot(N*pcond[5 * N // 12], color=c1)
        elif i == 1:
            ax6.plot(N * pcond[3 * N // 12], color=c2)
            ax6.plot(N * pcond[5 * N // 12], color=c1)
        elif i == 2:
            ax7.plot(N * pcond[3 * N // 12], color=c2)
            ax7.plot(N * pcond[5 * N // 12], color=c1)

    for i, axi in enumerate([ax2, ax3, ax4]):
        axi.axvline(5 * N / 12, color=c1)
        axi.axvline(3 * N / 12, color=c2)
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.set_yticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        axi.set_yticklabels([])
        axi.set_title(r'$%.f$' % Ecouplelst[i], fontsize=10)
        axi.tick_params(axis='both', labelsize=8)
        axi.yaxis.offsetText.set_fontsize(8)
        axi.set_xlabel(r'$x \ (\rm rev)$', fontsize=10)
        axi.set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        axi.set_xticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])

    for axi in [ax5, ax6, ax7]:
        axi.spines['right'].set_visible(False)
        axi.spines['top'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.set_xlabel(r'$y \ (\rm rev)$', fontsize=10)
        axi.set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        axi.set_xticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])
        axi.tick_params(axis='both', labelsize=8)
        axi.yaxis.offsetText.set_fontsize(8)
        axi.set_xlim((0, N))
        axi.set_ylim((0, 21))
        axi.set_yticklabels([])

    ax5.set_yticklabels(['$0$', '$10$', '$20$'])
    ax2.set_title(r'$\beta E_{\rm couple} = %.f$' % Ecouplelst[0], fontsize=10)
    ax2.set_yticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])
    ax5.set_ylabel(r'$p_{\rm ss}(y|x)$', fontsize=10)
    
    # Learning rate plot
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, 
                                          Ecouple_array_quad)))
    learning_rate = zeros(Ecouple_array_tot.size)
    for ii, Ecouple in enumerate(Ecouple_array_tot):
        input_file_name = target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                          "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                          "_outfile.dat"
        try:
            data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
            learning_rate[ii] = data_array[6]
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

    ax1.axvline(Ecouplelst[0], color='gray', linestyle='--')
    ax1.axvline(Ecouplelst[1], color='gray', linestyle='--')
    ax1.axvline(Ecouplelst[2], color='gray', linestyle='--')
    ax1.plot(Ecouple_array_tot, learning_rate, color='C6', linestyle='-', marker='o')

    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.yaxis.offsetText.set_fontsize(10)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.set_ylabel(r'$-\dot{I}_{\rm X} \, (\rm s^{-1})$', fontsize=10)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim([7*10**(-2), None])
    # ax1.set_title(r'$\rm Information \ flow$')
    ax1.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=10)

    # colorbar
    axax.set_title(r'$p_{\rm ss}(y|x)$')
    cbar = f1.colorbar(
        cs, cax=axax, orientation='horizontal', ax=ax2,
    )
    cbar.formatter.set_scientific(True)
    cbar.ax.tick_params(labelsize=8)
    # cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    f1.text(0.07, 0.99, r'$\rm a)$', fontsize=10)
    f1.text(0.07, 0.48, r'$\rm b)$', fontsize=10)
    f1.text(0.4, 0.48, r'$\rm c)$', fontsize=10)
    f1.text(0.7, 0.48, r'$\rm d)$', fontsize=10)
    f1.text(0.07, 0.25, r'$\rm e)$', fontsize=10)
    f1.text(0.4, 0.25, r'$\rm f)$', fontsize=10)
    f1.text(0.7, 0.25, r'$\rm g)$', fontsize=10)

    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2), bbox_inches='tight')


def plot_2D_prob_rot(target_dir):
    output_file_name1 = (
            target_dir + "results/" +
            "Pss_2D_rot_" + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_.pdf")

    Ecouplelst = array([2.0, 16.0, 128.0])
    Ecouplelabels = [r'$2$', r'$16$', r'$128$']
    phi = 0.0
    plt.figure()
    f1, ax1 = plt.subplots(2, 3, figsize=(6, 4), sharey='all')

    # Find max prob. to set plot range
    input_file_name = (
            "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
            "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
    try:
        data_array = loadtxt(
            input_file_name.format(E0, Ecouplelst[-1], E1, psi_1, psi_2, num_minima1, num_minima2, phi),
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
        flux_array = array(flux_array) / (dx * dx)

        dflux_array = empty((2, N, N))
        derivative_flux(flux_array, dflux_array, N, dx)
    except OSError:
        print('Missing file')
        print(input_file_name.format(E0, Ecouplelst[-1], E1, psi_1, psi_2, num_minima1, num_minima2, phase_array[0]))

    prob_max = amax(prob_ss_array)
    print(prob_max)

    # plots
    for ii, Ecouple in enumerate(Ecouplelst):
        input_file_name = (
                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set" +
                "/reference_" + "E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + "_outfile.dat")
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                usecols=(0, 2, 3, 4, 5, 6, 7, 8))

            N = int(sqrt(len(data_array[:, 0])))  # check grid size
            prob_ss_array = data_array[:, 0].reshape((N, N))

        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

        cs = ax1[0, ii].contourf(prob_ss_array.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)

        prob_new = zeros((N, N))
        for i in range(N):
            for j in range(N):
                # if (j < i and j + i < N) or (j > i and N < (j + i)):
                prob_new[i, (j + int(N / 2)) % N] = prob_ss_array[(i + j) % N, (i - j) % N]

        cs = ax1[1, ii].contourf(prob_new.T, cmap=plt.cm.cool, vmin=0, vmax=prob_max)

        ax1[1, ii].set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        ax1[0, ii].set_xticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        ax1[0, ii].set_yticks([0, N / 6, N / 3, N / 2, 2 * N / 3, 5 * N / 6, N])
        ax1[1, ii].set_xticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])
        ax1[0, ii].set_xticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])
        ax1[0, ii].set_yticklabels(['$0$', '', r'$\frac{1}{3}$', '', r'$\frac{2}{3}$', '', '$1$'])

        if ii == 0:
            ax1[0, ii].set_title(r"$\beta E_{\rm couple}$" + "={}".format(Ecouplelabels[ii]))
        else:
            ax1[0, ii].set_title(Ecouplelabels[ii])
        ax1[0, 0].set_ylabel(r'$y \ (\rm rev)$')
        ax1[1, 0].set_ylabel(r'$v \ (\rm rev)$')

        ax1[0, ii].set_xlabel(r'$x \ (\rm rev)$')
        ax1[1, ii].set_xlabel(r'$u \ (\rm rev)$')

    cax = f1.add_axes([0.96, 0.09, 0.03, 0.8])
    cbar = f1.colorbar(
        cs, cax=cax, orientation='vertical', ax=ax1
    )
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    f1.text(0.92, 0.65, r'$p_{\rm ss}(x, y)$', rotation=270, fontsize=12)
    f1.text(0.92, 0.22, r'$p_{\rm ss}(u, v)$', rotation=270, fontsize=12)

    plt.subplots_adjust(hspace=0.4)
    f1.savefig(output_file_name1.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), bbox_inches='tight')


def plot_EPR_cm_diff_Ecouple(target_dir):
    phase_shift = 0.0
    gamma = 1000
    barrier_height = [0.0, 2.0]
    lines = ['dashed', 'solid']

    output_file_name = (target_dir + "results/" +
                        "Entropy_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phase_{6}" + ".pdf")
    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(5, 4))

    for i, E0 in enumerate(barrier_height):
        E1 = E0
        # calculate entropy production
        Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))

        integrate_entropy_X = empty(Ecouple_array_tot.size)
        integrate_entropy_Y = empty(Ecouple_array_tot.size)
        integrate_entropy_sum = empty(Ecouple_array_tot.size)
        integrate_entropy_diff = empty(Ecouple_array_tot.size)

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if E0 == 0.0:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/211003/" +
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
                flux_array = array(flux_array)

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
        ax.plot(Ecouple_array_tot, integrate_entropy_sum, marker='o', color='C8', linestyle=lines[i])
        ax.plot(Ecouple_array_tot, integrate_entropy_diff, marker='o', color='C9', linestyle=lines[i])

        ax.set_xlim((2, None))
        ax.set_ylim((10, 6*10**2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
        ax.set_ylabel(r'$\dot{\Sigma} \ \rm (s^{-1})$', fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.yaxis.offsetText.set_fontsize(14)

    f.text(0.15, 0.35, r'$\dot{\Sigma}_{\rm \bar{X}}$', fontsize=14, color='C8')
    f.text(0.35, 0.75, r'$\dot{\Sigma}_{\rm \Delta X}$', fontsize=14, color='C9')
    f.legend(handles=[Line2D([0], [0], color='black', linestyle='dashed', lw=2, label=r'$0$'),
                      Line2D([0], [0], color='black', linestyle='solid', lw=2, label=r'$2$')],
             loc=[0.75, 0.7], frameon=False, fontsize=14, ncol=1, title=r'$\beta E^{\ddagger}$', title_fontsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), bbox_inches='tight')


def plot_super_grid_peak(target_dir):  # grid of plots of output power, entropy rate, learning rate
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([2, 4, 8])
    phi = 0.0
    power_x = zeros((Ecouple_array_tot.size, psi1_array.size))
    power_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    heat_x = zeros((Ecouple_array_tot.size, psi1_array.size))
    heat_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    power_x_to_y = zeros((Ecouple_array_tot.size, psi1_array.size))
    information_flow = zeros((Ecouple_array_tot.size, psi1_array.size))

    colorlst = ['C7', 'C8', 'C9']
    labellst = [r'$2$', r'$4$', r'$8$']

    output_file_name = (
            target_dir + "results/" + "Super_grid_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_log_.pdf")

    f, axarr = plt.subplots(3, 3, sharex='all', figsize=(8, 6.5))

    # Barrier data
    for k, ratio in enumerate(psi_ratio):
        for i, psi_1 in enumerate(psi1_array):
            psi_2 = - psi_1 / ratio
            print(psi_2)
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

            for j in range(3):
                axarr[j, i].fill_between([Ecouple_array_tot[idx - 1], Ecouple_array_tot[idx + 1]], -30, 10 ** 3,
                                         facecolor=colorlst[k], alpha=0.4)

            axarr[2, i].axhline(0, color='black')

            axarr[0, i].plot(Ecouple_array_tot, -power_y[:, i], 'o-', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[1, i].plot(Ecouple_array_tot, -power_x_to_y[:, i] - information_flow[:, i], 'o-', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[2, i].plot(Ecouple_array_tot, ((-heat_x[:, i] + information_flow[:, i]) - (-heat_y[:, i] - information_flow[:, i])),
                             'o-', color=colorlst[k], label=labellst[k], markersize=6)
            #axarr[3, i].plot(Ecouple_array_tot, information_flow[:, i], 'o-', color=colorlst[k], label=labellst[k],
            #                 markersize=6)

            for j in range(3):
                axarr[j, i].yaxis.offsetText.set_fontsize(14)
                axarr[j, i].tick_params(axis='y', labelsize=14)
                axarr[j, i].tick_params(axis='x', labelsize=14)
                axarr[j, i].set_xscale('log')
                axarr[j, i].set_yscale('log')
                axarr[j, i].spines['right'].set_visible(False)
                axarr[j, i].spines['top'].set_visible(False)
                axarr[j, i].set_xlim((2, 150))

            axarr[0, i].set_ylim((0.2, 120))
            axarr[1, i].set_ylim((0.9, 300))
            axarr[2, i].set_yscale('linear')
            # axarr[2, i].set_ylim((10**(-2), 10**2))
            axarr[2, i].spines['bottom'].set_visible(False)

            axarr[0, i].set_title(r'$%.0f$' % psi1_array[i], fontsize=18)

    axarr[2, 0].set_ylim((-0.5, 0.5))
    axarr[2, 1].set_ylim((-5, 2))
    axarr[2, 2].set_ylim((-22, 2))
    axarr[0, 1].set_yticklabels([])
    axarr[0, 2].set_yticklabels([])
    axarr[1, 1].set_yticklabels([])
    axarr[1, 2].set_yticklabels([])

    axarr[0, 0].set_ylabel(r'$-\beta \mathcal{P}_{\rm Y} \ \rm (s^{-1})$', fontsize=14)
    axarr[1, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X} \ \rm (s^{-1})$', fontsize=14)
    axarr[2, 0].set_ylabel(r'$\dot{\Sigma}_{\rm X} - \dot{\Sigma}_{\rm Y} \ \rm (s^{-1})$', fontsize=14)
    # axarr[3, 0].set_ylabel(r'$\dot{I}_{\rm Y} \ \rm (s^{-1})$', fontsize=14)

    # f.legend(handles=[Line2D([0], [0], color=colorlst[0], linestyle='solid', lw=2, label=labellst[0], marker='o'),
    #                   Line2D([0], [0], color=colorlst[1], linestyle='solid', lw=2, label=labellst[1], marker='o'),
    #                   Line2D([0], [0], color=colorlst[2], linestyle='solid', lw=2, label=labellst[2], marker='o')],
    #          title=r'$-\mu_{\rm X}/\mu_{\rm Y}$', fontsize=18, ncol=3, frameon=False,
    #          bbox_to_anchor=(0.75, 1.14), title_fontsize=18)

    # f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    axarr[0, 2].legend(title=r'$-\mu_{\rm X}/\mu_{\rm Y}$', ncol=1, frameon=False, fontsize=12, title_fontsize=12,
                       loc=[0.6, 0.03])

    f.text(0.5, 0.95, r'$\beta \mu_{\rm X}\, (\rm rad^{-1})$', ha='center', fontsize=18)
    f.text(0.5, 0.03, r'$\beta E_{\rm couple}$', ha='center', fontsize=18)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi), bbox_inches='tight')


def plot_power_ratio_Ecouple(input_dir):  # plot power and efficiency as a function of the coupling strength
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))
    phi = 0.0
    learning_rate = zeros(Ecouple_array_tot.size)
    transmitted_power = zeros(Ecouple_array_tot.size)
    input_power = zeros(Ecouple_array_tot.size)
    output_power = zeros(Ecouple_array_tot.size)

    f, axarr = plt.subplots(1, 1, sharey='row', figsize=(6, 4))

    output_file_name = input_dir + "results/" + \
                       "Power_ratio_Ecouple_E0_{0}_E1_{1}_psi1_{2}_phi_{3}_zoom.pdf"

    for j, psi_2 in enumerate([-1.0, -0.5, -0.25]):
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                              "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                              "_outfile.dat"
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                input_power[ii] = data_array[1]
                output_power[ii] = data_array[2]
                transmitted_power[ii] = data_array[5]
                learning_rate[ii] = -data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        ratio = -input_power * output_power / (transmitted_power + learning_rate) ** 2
        axarr.plot(Ecouple_array_tot, ratio, linestyle='-', marker='o', label=psi_2)

        # dratio = zeros(len(ratio) - 1)
        #
        # for ii in range(len(Ecouple_array_tot) - 1):
        #     dratio[ii] = (ratio[ii + 1] - ratio[ii]) / (Ecouple_array_tot[ii + 1] - Ecouple_array_tot[ii])
        #
        # for ii in range(len(dratio)-1, 0, -1):
        #     if dratio[ii] < 0:
        #         print(ii)
        #         break

    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(12)
    axarr.tick_params(axis='both', labelsize=12)
    axarr.set_ylabel(r'$\frac{-\mathcal{P}_{\rm X} \mathcal{P}_{\rm Y}}{(\mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X})^2}$', fontsize=14)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')
    axarr.legend(frameon=False, title=r'$\mu_{\rm Y}$')
    axarr.set_ylim([0, 1.15])

    axarr.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, phi), bbox_inches='tight')


def plot_power_bound_EPR(target_dir):
    phase_shift = 0.0
    barrier_height = array([0.0, 2.0])
    lines = ['dashed', 'solid']
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Power_bound_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(2, 1, figsize=(5, 8))
    for j, E0 in enumerate(barrier_height):
        E1 = E0
        if E0 == 0.0:
            Ecouple_array_tot = sort(
                concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad, Ecouple_array_peak)))
        else:
            Ecouple_array_tot = sort(
                concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak, Ecouple_array_quad)))

        power_x = empty(Ecouple_array_tot.size)
        power_y = empty(Ecouple_array_tot.size)
        energy_xy = empty(Ecouple_array_tot.size)
        learning_rate = empty(Ecouple_array_tot.size)

        for i, Ecouple in enumerate(Ecouple_array_tot):
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                power_x[i] = data_array[1]
                power_y[i] = data_array[2]
                energy_xy[i] = data_array[5]
                learning_rate[i] = data_array[6]
            except OSError:
                print('Missing file 1')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        ax[0].axhline(0, color='black')
        # ax.axhline(1, color='grey')

        if j == 0:
            ax[0].plot(Ecouple_array_tot, power_x, linestyle=lines[j], marker='o', color='tab:blue')
            ax[0].plot(Ecouple_array_tot, -power_y, linestyle=lines[j], marker='o', color='tab:purple')
            ax[0].plot(Ecouple_array_tot, -energy_xy - learning_rate, linestyle=lines[j], marker='o',
                    color='black')
        else:
            ax[0].plot(Ecouple_array_tot, power_x, linestyle=lines[j], marker='o', color='tab:blue')
            ax[0].plot(Ecouple_array_tot, -power_y, linestyle=lines[j], marker='o', color='tab:purple')
            ax[0].plot(Ecouple_array_tot, -energy_xy - learning_rate, linestyle=lines[j], marker='o', color='black')

    ax[0].set_ylim((7, 3 * 10 ** 2))
    ax[0].set_xlim((2, None))
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\beta \mathcal{P} \ (\rm s^{-1})$', fontsize=14)
    ax[0].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)

    # EPR
    for i, E0 in enumerate(barrier_height):
        E1 = E0
        if E0 == 0.0:
            Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_quad, Ecouple_array_peak)))
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
                print('Missing file 2')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        ax[1].plot(Ecouple_array_tot, -heat_x + learning_rate, linestyle=lines[i], marker='o', color='tab:orange')
        ax[1].plot(Ecouple_array_tot, -heat_y - learning_rate, linestyle=lines[i], marker='o', color='tab:red')

    ax[1].set_xlim((2, None))
    ax[1].set_ylim((3, 3*10**2))
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax[1].set_ylabel(r'$\dot{\Sigma} \, (\rm s^{-1})$', fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].yaxis.offsetText.set_fontsize(12)

    f.text(0.32, 0.85, r'$\beta \mathcal{P}_{\rm X}$', fontsize=14, color='tab:blue')
    f.text(0.14, 0.67, r'$\beta \mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X}$', fontsize=14, color='black')
    f.text(0.44, 0.55, r'$-\beta \mathcal{P}_{\rm Y}$', fontsize=14, color='tab:purple')
    f.text(0.35, 0.4, r'$\dot{\Sigma}_{\rm X}$', fontsize=14, color='tab:orange')
    f.text(0.2, 0.25, r'$\dot{\Sigma}_{\rm Y}$', fontsize=14, color='tab:red')
    f.text(0.05, 0.87, r'$\rm a)$', fontsize=14)
    f.text(0.05, 0.45, r'$\rm b)$', fontsize=14)

    f.legend(handles=[Line2D([0], [0], color='gray', linestyle='dashed', lw=2, label=r'$0$'),
                      Line2D([0], [0], color='gray', linestyle='solid', lw=2, label=r'$2$')],
             loc=[0.77, 0.35], frameon=False, fontsize=14, ncol=1, title=r'$\beta E^{\ddagger}$', title_fontsize=14)

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), bbox_inches='tight')


def plot_alt_efficiencies_Ecouple(target_dir):
    phase_shift = 0.0
    barrier_height = array([2.0])
    linestyles = ['solid']
    input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + "power_heat_info_" +
                       "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
    output_file_name = (target_dir + "results/" + "Information_engine-ness_Ecouple_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phi_{6}" + "_.pdf")

    plt.figure()
    f, ax = plt.subplots(1, 1, figsize=(5, 4))
    for j, E0 in enumerate(barrier_height):
        E1 = E0
        if E0 == 0.0:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double)))
        else:
            Ecouple_array_total = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_array_peak)))

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

        # ax.plot(Ecouple_array_total, -(energy_xy + learning_rate)/power_x, linestyle=linestyles[j], marker='o',
        #            color='tab:blue', label=r'$\eta_{\rm X}$')
        # ax.plot(Ecouple_array_total, power_y/(energy_xy + learning_rate), linestyle=linestyles[j], marker='o',
        #            color='tab:orange', label=r'$\eta_{\rm Y}$')
        ax.plot(Ecouple_array_total, -learning_rate / (energy_xy - learning_rate), linestyle=linestyles[j], marker='o',
                   color='tab:blue')

    # ax.set_ylim((7, 3 * 10 ** 2))
    # ax.set_xlim((2, None))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylabel(r'$\eta$', fontsize=14)
    ax.set_ylabel(r'$\frac{\dot{I}_{\rm X}}{\mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X}}$', fontsize=14)
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    # ax.legend(frameon=False)

    # f.text(0.32, 0.85, r'$\mathcal{P}_{\rm X}$', fontsize=14, color='tab:blue')
    # f.text(0.14, 0.65, r'$\mathcal{P}_{\rm X \to Y} + \dot{I}_{\rm X}$', fontsize=14, color='black')

    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift), bbox_inches='tight')


if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    # plot_energy_flow(target_dir)
    # plot_entropy_production_Ecouple(target_dir)
    # plot_power_bound_Ecouple(target_dir)
    # plot_nn_learning_rate_Ecouple(target_dir)
    # plot_nn_learning_rate_Ecouple_inset(target_dir)
    # plot_power_entropy_correlation(target_dir)
    # plot_2D_prob_triple(target_dir)
    # plot_lr_prob_slice(target_dir)
    # plot_2D_prob_rot(target_dir)
    plot_EPR_cm_diff_Ecouple(target_dir)
    # plot_super_grid_peak(target_dir)
    # plot_power_ratio_Ecouple(target_dir)
    # plot_power_bound_EPR(target_dir)
    # plot_alt_efficiencies_Ecouple(target_dir)
