from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz, log, argmax, sin, argmin, \
    sort, concatenate, ones
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from utilities import step_probability_X, calc_derivative_pxgy, step_probability_Y, calc_flux_2
from ATP_energy_transduction import derivative_flux

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 540  # N x N grid is used for Fokker-Planck simulations
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
Ecouple_array_double = array([1.41, 2.83, 5.66, 11.31, 22.63, 45.25, 90.51])
Ecouple_extra = array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0])
Ecouple_array_quad = array([1.19, 1.68, 2.38, 3.36, 4.76, 6.73, 9.51, 13.45, 19.03, 26.91, 38.05, 53.82, 76.11, 107.63])
Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_extra)))


def calc_marg_derivative(flux_array, dflux_array, N, dx):
    # explicit update of the corners
    dflux_array[0] = (flux_array[1] - flux_array[N - 1]) / (2.0 * dx)
    dflux_array[N - 1] = (flux_array[0] - flux_array[N - 2]) / (2.0 * dx)

    # for points with well defined neighbours
    for j in range(1, N - 1):
        dflux_array[j] = (flux_array[j + 1] - flux_array[j - 1]) / (2.0 * dx)


def calc_derivative(flux_array, dflux_array, N, dx, k):
    if k == 0:
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


def plot_ITQ_Ecouple(target_dir, quantity, dt):  # grid of plots of the flux as a function of the phase offset
    Barrier_heights = [2.0]
    phi = 0.0

    if quantity == 'nostalgia':
        output_file_name = (
                target_dir + "results/" + "Nostalgia_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'learning_rate':
        output_file_name = (
                target_dir + "results/" + "LearningRate_test_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'learning_rate_2':
        output_file_name = (
                target_dir + "results/" + "LearningRate2_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'learning_rate_3':
        output_file_name = (
                target_dir + "results/" + "LearningRate3_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'mutual_info':
        output_file_name = (
                target_dir + "results/" + "MutualInfo_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'relative_entropy':
        output_file_name = (
                target_dir + "results/" + "RelEntropy_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")

    f, ax = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))

    for jj, E0 in enumerate(Barrier_heights):
        E1 = E0
        if E0 == 0.0:
            information = zeros(Ecouple_array.size)
            Ecouple_array_tot = Ecouple_array
        elif E0 == 2.0:
            Ecouple_array_tot = array(
                [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
                 45.25, 64.0, 90.51, 128.0])
            information = zeros(Ecouple_array_tot.size)

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if E1 == 0.0:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            elif E1 == 2.0:
                if Ecouple in Ecouple_array:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
                elif Ecouple in array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0]):
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_Extra_measurements_Ecouple/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                    usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))
                dx = 2 * math.pi / N
                positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                prob_eq_array = data_array[:, 1].T.reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            if quantity == 'nostalgia':
                step_X = empty((N, N))
                step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                # instantaneous predictive power
                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

                information[ii] = timescale*(Imem - Ipred) / dt

            elif quantity == 'learning_rate':
                flux_array = empty((2, N, N))
                calc_flux(positions, prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

                dflux_array = empty((2, N, N))
                derivative_flux(flux_array, dflux_array, N, dx)

                learning = dflux_array[1, ...] * log(prob_ss_array.sum(axis=0)/prob_ss_array)

                information[ii] = trapz(trapz(learning)) * timescale

            elif quantity == 'learning_rate_2':
                step_X = empty((N, N))
                step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

                step_Y = empty((N, N))
                step_probability_Y(step_Y, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

                flux_array = empty((2, N, N))
                calc_flux(positions, prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

                dflux_array_y = empty((N, N))
                calc_derivative(flux_array[1, ...].reshape((N, N)), dflux_array_y, N, dx, 1)
                dflux_array_x = empty((N, N))
                calc_derivative(flux_array[0, ...].reshape((N, N)), dflux_array_x, N, dx, 0)

                marg2 = dflux_array_x * (log(prob_ss_array.sum(axis=1)) + 1)  # 2 -> 1
                cond2 = dflux_array_x * (log(step_X / step_X.sum(axis=0)) + 1)

                marg = dflux_array_y * (log(prob_ss_array.sum(axis=0)) + 1)  # 1 -> 2
                cond = dflux_array_y * (log(step_Y / step_Y.sum(axis=1)) + 1)

                learning = marg  # l 1->2 = - l 2->1 at steady state
                learning2 = -cond
                learning3 = marg2
                learning4 = -cond2

                information[0, ii] = trapz(trapz(learning, dx=1), dx=1)
                information[1, ii] = trapz(trapz(learning2, dx=1), dx=1)
                information[2, ii] = trapz(trapz(learning3, dx=1), dx=1)
                information[3, ii] = trapz(trapz(learning4, dx=1), dx=1)

            elif quantity == 'learning_rate_3':
                denergy = empty((N, N))
                for i in range(N):
                    for j in range(N):
                        denergy[i, j] = - 0.5 * Ecouple * sin(positions[i] - positions[j]) #\
                                       # + 1.5 * E1 * sin(3 * positions[j])

                dP = empty((N, N))
                calc_derivative(prob_ss_array, dP, N, dx, 1)

                dPy = empty((N, N))
                calc_marg_derivative(prob_ss_array.sum(axis=0), dPy, N, dx)

                learning = prob_ss_array * (denergy + dP/prob_ss_array) * ((dPy/prob_ss_array.sum(axis=0))[None, :] - dP/prob_ss_array)

                information[ii] = 10**(-3) * trapz(trapz(learning, dx=1), dx=1)

            elif quantity == 'mutual_info':
                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                information[ii] = Imem

            elif quantity == 'relative_entropy':
                information[ii] = (prob_ss_array * log(prob_ss_array/prob_eq_array)).sum(axis=None)

        ax.axhline(0, color='black')
        if E0 == 0.0:
            ax.plot(Ecouple_array_tot, information, 'o', color='C0', label='$0$', markersize=8)
        elif E0 == 2.0:
            ax.plot(Ecouple_array_tot, information, 'o', color='C1', label=r'$2$', markersize=8)
            # ax.plot(Ecouple_array_tot, information[0], 'o', color='C1', label=r'$d_t S[\theta_1(t)]$', markersize=8)
            # ax.plot(Ecouple_array_tot, information[1], 'o', color='C2',
            #         label=r'$-\partial_{\tau} S[\theta_{\rm 1}(t + \tau)| \theta_{\rm o}(t)]$', markersize=8)
            # ax.plot(Ecouple_array_tot, information[2], 'o', color='C3',
            #         label=r'$d_t S[\theta_{\rm o}(t)]$', markersize=8)
            # ax.plot(Ecouple_array_tot, information[3], 'o', color='C4',
            #         label=r'$-\partial_{\tau} S[\theta_{\rm o}(t + \tau)| \theta_1(t)]$', markersize=8)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim((-0.1, 2.5))
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    if quantity == 'nostalgia' or 'learning_rate' or 'learning_rate_2' or 'learning_rate_3':
        # ax.set_ylabel(r'$\rm d_{\tau} S[F_{\rm o}(t + \tau) | F_1(t)]$', fontsize=20)
        ax.set_ylabel(r'$\ell_{\rm F_1} (\rm nats/s)$', fontsize=20)
    elif quantity == 'mutual_info':
        ax.set_ylabel(r'$I(\theta_{\rm o}(t), \theta_1(t))$', fontsize=20)
    elif quantity == 'relative_entropy':
        ax.set_ylabel(r'$\mathcal{D}_{\rm KL}( P_{\rm ss} || P_{\rm eq} )$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # leg = ax.legend(fontsize=16, loc='best', frameon=False)
    leg = ax.legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    # f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_MI_Ecouple(target_dir, dt):
    Barrier_heights = [2.0]
    phi = 0.0
    timestep = array([dt/100, dt/10, dt/2, dt, 2*dt, 10*dt, 100*dt])

    output_file_name = (
            target_dir + "results/" + "LearningRate_Ecouple_dt_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")

    f, ax = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))

    for jj, E0 in enumerate(Barrier_heights):
        E1 = E0
        Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_extra)))
        information = zeros((Ecouple_array_tot.size, timestep.size))

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if E1 == 0.0:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            elif E1 == 2.0:
                if Ecouple in Ecouple_extra:
                    input_file_name = (
                                "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200511_2kT_extra/" +
                                "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                "_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200427_strongforces/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                    usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))
                dx = 2 * math.pi / N
                positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                prob_eq_array = data_array[:, 1].T.reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            for j in range(N):
                for i in range(N):
                    if prob_ss_array[i, j] == 0.0:
                        prob_ss_array[i, j] = 10e-18

            for j, dt in enumerate(timestep):
                step_X = empty((N, N))
                step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)

                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                # instantaneous predictive power
                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)
                information[ii, j] = timescale * (Imem - Ipred) / dt

        # ax.axhline(0, color='black')
        for j, dt in enumerate(timestep):
            ax.plot(Ecouple_array_tot, information[:, j], 'o-', label=dt, markersize=8)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim((0, None))
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    ax.set_ylabel(r'$\ell_1$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    leg = ax.legend(fontsize=16, loc='best', frameon=False, title='$\Delta t$')
    # leg = ax.legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    # f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_learning_rates_Ecouple(target_dir):
    phi = 0.0
    barrier_heights = [2.0]
    gridpoints = ['360', '720']

    output_file_name = (target_dir + "results/" + "LearningRate_Ecouple_N_" +
                        "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")

    f, ax = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))
    ax.axhline(0, color='black')

    for i, points in enumerate(gridpoints):
        Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_extra)))

        if i == 0:
            input_file_name = (
                    target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                    "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                    "_outfile.dat")
        else:
            input_file_name = (
                    target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                    "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                    "_outfile_N720.dat")

        learning_rate = zeros(Ecouple_array_tot.size)
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))
                learning_rate[ii] = data_array[6]

            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

        ax.plot(Ecouple_array_tot, learning_rate, 'o-', markersize=8, label=points)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim((10e-1, 10e1))
    ax.set_ylim((0, None))
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    ax.set_ylabel(r'$\ell_1 (\rm nats/s)$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    leg = ax.legend(fontsize=16, loc='best', frameon=False)
    leg = ax.legend(title=r'N', fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_nostalgia_Ecouple_grid(target_dir):  # grid of plots of the flux as a function of the phase offset
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
                                                      Ecouple_extra)))
            elif abs(psi_2) > psi_1/2:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, array([11.31, 22.63, 45.25, 90.51]))))
            else:
                Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double, Ecouple_extra)))

            information = ones(Ecouple_array_tot.size)*(-10)

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" +
                                   "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                                   "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple), usecols=6)
                    information[ii] = data_array
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

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
    f.text(0.05, 0.5, r'$\ell_{\rm o \to 1} (\rm nats/s)$', va='center', rotation='vertical', fontsize=24)
    f.text(0.5, 0.95, r'$-\mu_{\rm H^+} / \mu_{\rm ATP}$', ha='center', rotation=0, fontsize=24)
    f.text(0.95, 0.5, r'$\mu_{\rm H^+}\ (k_{\rm B} T / \rm rad)$', va='center', rotation=270, fontsize=24)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi))


def plot_correlation_nostalgia_power_peaks(target_dir):
    output_file_name = (
            target_dir + "results/" + "Nostalgia_power_peak_correlation_" +
            "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")

    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    phi = 0.0

    information = zeros((Ecouple_array_tot.size, psi1_array.size, psi_ratio.size))
    output_power = zeros((Ecouple_array_tot.size, psi1_array.size, psi_ratio.size))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio

            # Calculate information
            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if ((psi_1 == 4.0 and (psi_2 == -1.0 or psi_2 == -2.0)) or (psi_1 == 8.0) or
                        (psi_1 == 2.0 and psi_2 == -1.0)) and Ecouple in Ecouple_array:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")
                else:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                       "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                       "_outfile.dat")

                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                        usecols=(0, 3, 4, 5, 6, 7, 8))
                    N = int(sqrt(len(data_array)))  # check grid size
                    prob_ss_array = data_array[:, 0].T.reshape((N, N))
                    drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

                step_X = empty((N, N))
                step_probability_X(
                    step_X, prob_ss_array, drift_at_pos, diffusion_at_pos,
                    N, dx, 0.001
                )

                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                # instantaneous predictive power
                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

                information[ii, i, j] = Imem - Ipred

    # Grab output power
    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                input_file_name = (target_dir + "plotting_data/" + "flux_power_efficiency_"
                                   + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=4)
                    if psi_1 == 4.0 and psi_2 == -2.0 and Ecouple in Ecouple_array:
                        power_y = data_array[0]
                    else:
                        power_y = data_array
                    output_power[ii, i, j] = -power_y
                except OSError:
                    print('Missing file power')

    # Plot correlation
    f, ax = plt.subplots(1, 1, figsize=(6, 6))

    # get argmax from each array
    maxnos_pos = argmax(information, axis=0)
    maxpow_pos = argmax(output_power, axis=0)

    print(maxnos_pos)
    print(maxpow_pos)

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            ax.plot(Ecouple_array_tot[maxpow_pos[i, j]], Ecouple_array_tot[maxnos_pos[i, j]], 'o', color='C1')

    ax.set_xlabel(r'$\beta E^{\rm max\ power}_{\rm couple}$', fontsize=20)
    ax.set_ylabel(r'$\beta E^{\rm max\ nostalgia}_{\rm couple}$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_ITQ_phi(target_dir, quantity, dt):
    # Ecouple_array_tot = array(
    #     [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
    #      45.25, 64.0, 90.51, 128.0])
    Ecouple_array_tot = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    # Ecouple_array_tot = array([2.0])
    phi_array = array([0.0, 0.175, 0.349066, 0.524, 0.698132, 0.873, 1.0472, 1.222, 1.39626, 1.571, 1.74533, 1.92,
                       2.0944])

    for ii, Ecouple in enumerate(Ecouple_array_tot):

        if quantity == 'nostalgia':
            output_file_name = (
                    target_dir + "results/" + "Nostalgia_phi_"
                    + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_Ecouple_{6}" + "_.pdf")

        f, ax = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))
        ax.axhline(0, color='black')

        # Fokker-Planck results (barriers)
        information = zeros(phi_array.size)

        for j, phi in enumerate(phi_array):

            if Ecouple in Ecouple_array and phi in array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944]):
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            else:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")

            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                    usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))  # check grid size
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                prob_eq_array = data_array[:, 1].T.reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            if quantity == 'nostalgia':
                step_X = empty((N, N))
                step_probability_X(
                    step_X, prob_ss_array, drift_at_pos, diffusion_at_pos,
                    N, dx, dt
                )

                # instantaneous memory
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                # instantaneous predictive power
                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

                information[j] = timescale*(Imem - Ipred)/dt

        ax.plot(phi_array, information, 'o', color='C1', label='$2$', markersize=8)

        ax.yaxis.offsetText.set_fontsize(14)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel(r'$n \phi$ (\rm rev)', fontsize=20)
        if quantity == 'nostalgia':
            ax.set_ylabel(r'$\ell_{\rm F_1} (\rm nats/s)$', fontsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3])
        ax.set_xticklabels(['$0$', '', '', '$1/2$', '', '', '$1$'])
        # ax.set_ylim((0, None))
        ax.set_xlim((0, 2 * pi/3))

        f.tight_layout()
        f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))


def plot_super_grid(target_dir, dt):  # grid of plots of output power, dissipation, MI, rel. entropy, learning rate
    # Ecouple_array_tot = array(
    #     [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0, 32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi2_array = array([-1.0, -2.0, -4.0])
    phi = 0.0
    output_power = zeros((Ecouple_array_tot.size, psi1_array.size))
    dissipation = zeros((Ecouple_array_tot.size, psi1_array.size))
    mutual_info = zeros((Ecouple_array_tot.size, psi1_array.size))
    learning_rate = zeros((Ecouple_array_tot.size, psi1_array.size))
    rel_entropy = zeros((Ecouple_array_tot.size, psi1_array.size))

    mutual_info_zero = zeros((Ecouple_array.size, psi1_array.size))
    learning_rate_zero = zeros((Ecouple_array.size, psi1_array.size))
    rel_entropy_zero = zeros((Ecouple_array.size, psi1_array.size))

    colorlst = ['C1', 'C9']
    labellst = ['$2$', '$4$']

    output_file_name = (
            target_dir + "results/" + "Super_grid_double_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_.pdf")

    f, axarr = plt.subplots(5, 3, sharex='all', figsize=(8, 10))

    # Barrier-less data
    for i, psi_1 in enumerate(psi1_array):
        psi_2 = psi2_array[i]
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/"
                               + "flux_zerobarrier_psi1_{0}_psi2_{1}_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(psi_1, psi_2))
                Ecouple_array2 = array(data_array[:, 0])
                flux_x_array = array(data_array[:, 1])
                flux_y_array = array(data_array[:, 2])
                power_y = -flux_y_array * psi_2 * 2 * pi * timescale
                power_x = flux_x_array * psi_1 * 2 * pi * timescale
                dissipation_zero = power_x - power_y
            except OSError:
                print('Missing file no barriers')

        for ii, Ecouple in enumerate(Ecouple_array):
            input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/" +
                               "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                               "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(0.0, Ecouple, 0.0, psi_1, psi_2, num_minima1, num_minima2, phi),
                    usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                prob_eq_array = data_array[:, 1].T.reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            except OSError:
                print('Missing file no barriers')
                print(input_file_name.format(0.0, Ecouple, 0.0, psi_1, psi_2, num_minima1, num_minima2, phi))

            # calculate mutual information
            mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
            Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

            mutual_info_zero[ii, i] = Imem

            # calculate learning rate (=nostalgia)
            step_X = empty((N, N))
            step_probability_X(
                step_X, prob_ss_array, drift_at_pos, diffusion_at_pos,
                N, dx, dt
            )

            mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
            Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

            pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
            Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

            learning_rate_zero[ii, i] = timescale*(Imem - Ipred)/dt

            # calculate relative entropy
            rel_entropy_zero[ii, i] = (prob_ss_array * log(prob_ss_array / prob_eq_array)).sum(axis=None)

            if Ecouple == 64.0:
                mutual_info_zero[5, 1] = -100
                mutual_info_zero[5, 2] = -100
                learning_rate_zero[5, 1] = -100
                learning_rate_zero[5, 2] = -100
                rel_entropy_zero[5, 1] = -100
                rel_entropy_zero[5, 2] = -100

        axarr[0, i].plot(Ecouple_array2, power_y, '-', color='C0', label='$0$', linewidth=2)
        axarr[1, i].plot(Ecouple_array2, dissipation_zero, '-', color='C0', label='$0$', linewidth=2)
        axarr[2, i].plot(Ecouple_array, mutual_info_zero[:, i], 'o', color='C0', label='$0$', markersize=8)
        axarr[3, i].plot(Ecouple_array, learning_rate_zero[:, i], 'o', color='C0', label='$0$', markersize=8)
        axarr[4, i].plot(Ecouple_array, rel_entropy_zero[:, i], 'o', color='C0', label='$0$', markersize=8)

    # Barrier data
    for k, E0 in enumerate([2.0, 4.0]):
        E1 = E0
        for i, psi_1 in enumerate(psi1_array):
            psi_2 = psi2_array[i]

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if E0 == 2.0:
                    if Ecouple in Ecouple_extra:
                        input_file_name = (
                                    "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/200511_2kT_extra/" +
                                    "flux_power_efficiency_" +
                                    "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/plotting_data/" +
                                           "flux_power_efficiency_" +
                                           "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                elif E0 == 4.0:
                    input_file_name = (
                                "/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/200506_4kTbarrier/spectral/" +
                                "flux_power_efficiency_" +
                                "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                        usecols=(3, 4))
                    if E0 == 2.0 and psi_1 == 4.0 and psi_2 == -2.0 and Ecouple in Ecouple_array:
                        power_x = data_array[0, 0]
                        power_y = data_array[0, 1]
                    else:
                        power_x = data_array[0]
                        power_y = data_array[1]
                except OSError:
                    print('Missing file power')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

                # grab output power
                output_power[ii, i] = -2 * pi * timescale * power_y
                # calculate dissipation
                dissipation[ii, i] = 2 * pi * timescale * (power_x + power_y)

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if E0 == 2.0:
                    if Ecouple in Ecouple_array:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    elif Ecouple in Ecouple_extra:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                elif E0 == 4.0:
                    if Ecouple in Ecouple_extra:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/spectral/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                        usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                    N = int(sqrt(len(data_array)))  # check grid size
                    prob_ss_array = data_array[:, 0].T.reshape((N, N))
                    prob_eq_array = data_array[:, 1].T.reshape((N, N))
                    drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

                # calculate mutual information
                for l in range(N):
                    for j in range(N):
                        if prob_ss_array[l, j] == 0.0:
                            prob_ss_array[l, j] = 1e-18
                mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
                Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

                mutual_info[ii, i] = Imem

                # calculate learning rate (=nostalgia)
                for l in range(N):
                    for j in range(N):
                        if step_X[l, j] == 0.0:
                            step_X[l, j] = 1e-18
                step_X = empty((N, N))
                step_probability_X(
                    step_X, prob_ss_array, drift_at_pos, diffusion_at_pos,
                    N, dx, dt
                )

                pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
                Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

                learning_rate[ii, i] = timescale*(Imem - Ipred)/dt

                # calculate relative entropy
                rel_entropy[ii, i] = (prob_ss_array * log(prob_ss_array / prob_eq_array)).sum(axis=None)

            # plot line at coupling strength corresponding to max power
            maxpos = argmax(output_power[:, i], axis=0)
            for j in range(5):
                axarr[j, i].axvline(Ecouple_array_tot[maxpos], linestyle='--', color=colorlst[k])

            axarr[0, i].plot(Ecouple_array_tot, output_power[:, i], 'o', color=colorlst[k], label=labellst[k], markersize=6)
            axarr[1, i].plot(Ecouple_array_tot, dissipation[:, i], 'o', color=colorlst[k], label=labellst[k], markersize=6)
            axarr[2, i].plot(Ecouple_array_tot, mutual_info[:, i], 'o', color=colorlst[k], label=labellst[k], markersize=6)
            axarr[3, i].plot(Ecouple_array_tot, learning_rate[:, i], 'o', color=colorlst[k], label=labellst[k], markersize=6)
            axarr[4, i].plot(Ecouple_array_tot, rel_entropy[:, i], 'o', color=colorlst[k], label=labellst[k], markersize=6)

            for j in range(5):
                axarr[j, i].yaxis.offsetText.set_fontsize(14)
                axarr[j, i].ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
                axarr[j, i].tick_params(axis='y', labelsize=14)
                axarr[j, i].set_xscale('log')
                axarr[j, i].spines['right'].set_visible(False)
                axarr[j, i].spines['top'].set_visible(False)
                axarr[j, i].set_xlim((2, 150))
                axarr[j, i].set_ylim(bottom=0)

            axarr[0, i].set_title(r'$%.0f$' % psi1_array[i], fontsize=18)

    axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1})$', fontsize=14)
    # axarr[1, 0].set_ylabel(r'$\beta (\mathcal{P}_{\rm H^+} -\mathcal{P}_{\rm ATP}) (\rm s^{-1}) $', fontsize=14)
    axarr[2, 0].set_ylabel(r'$I(\theta_{\rm o}(t), \theta_1(t)) (\rm nats)$', fontsize=14)
    axarr[2, 0].set_ylim(top=2.5)
    axarr[2, 1].set_ylim(top=2.5)
    axarr[2, 2].set_ylim(top=2.5)
    axarr[3, 0].set_ylabel(r'$\ell_{\rm F_1} (\rm nats/s)$', fontsize=14)
    axarr[3, 0].set_ylim(top=1.2)
    axarr[3, 1].set_ylim(top=2.6)
    axarr[3, 2].set_ylim(top=5.5)
    axarr[4, 0].set_ylabel(r'$\mathcal{D}_{\rm KL} ( P_{\rm ss} || P_{\rm eq} ) (\rm nats)$', fontsize=14)
    axarr[4, 0].set_ylim(top=0.5)
    axarr[4, 1].set_ylim(top=2)
    axarr[4, 2].set_ylim(top=5)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$\mu_{\rm H^+}$', ha='center', fontsize=20)
    f.text(0.5, 0.05, r'$E_{\rm couple}$', ha='center', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi))


def plot_super_grid_peak(target_dir, dt):  # grid of plots of output power, dissipation, MI, rel. entropy, learning rate
    # Ecouple_array_tot = array(
    #     [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 22.63, 24.0, 32.0, 45.25, 64.0,
         90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi2_array = array([-1.0, -2.0, -4.0])
    phi = 0.0
    output_power = zeros((Ecouple_array_tot.size, psi1_array.size))
    flux = zeros((Ecouple_array_tot.size, psi1_array.size))
    energy_flow = zeros((Ecouple_array_tot.size, psi1_array.size))
    learning_rate = zeros((Ecouple_array_tot.size, psi1_array.size))

    colorlst = ['C1', 'C9']
    labellst = ['$2$', '$4$']

    output_file_name = (
            target_dir + "results/" + "Super_grid_peak_l2_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_log_.pdf")

    f, axarr = plt.subplots(4, 3, sharex='all', figsize=(8, 8))

    # Barrier data
    for k, E0 in enumerate([2.0, 4.0]):
        E1 = E0
        for i, psi_1 in enumerate(psi1_array):
            psi_2 = psi2_array[i]

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if E0 == 2.0:
                    if Ecouple in Ecouple_array:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                    elif Ecouple in Ecouple_extra:
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT/" +
                                    "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                    "_outfile.dat")
                    else:
                        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                           "_outfile.dat")
                elif E0 == 4.0:
                    if Ecouple in Ecouple_extra:
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/6kT/" +
                                    "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                    "_outfile.dat")
                    else:
                        input_file_name = (
                                    "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200506_4kTbarrier/spectral/" +
                                    "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                    "_outfile.dat")
                try:
                    data_array = loadtxt(
                        input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                        usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                    N = int(sqrt(len(data_array)))  # check grid size
                    dx = 2 * math.pi / N  # spacing between gridpoints
                    positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
                    prob_ss_array = data_array[:, 0].T.reshape((N, N))
                    prob_eq_array = data_array[:, 1].T.reshape((N, N))
                    drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

                # calculate flux
                flux_array = zeros((2, N, N))
                calc_flux_2(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

                flux[ii, i] = trapz(trapz(flux_array[1, ...])) * timescale

                # calculate output power
                output_power[ii, i] = -psi_2 * flux[ii, i]

                # calculate learning rate (=nostalgia)
                dflux_array = empty((2, N, N))
                derivative_flux(flux_array, dflux_array, N, dx)

                learning = dflux_array[1, ...] * log(prob_ss_array.sum(axis=0)/prob_ss_array)

                learning_rate[ii, i] = trapz(trapz(learning)) * timescale

                # calculate energy flow
                # force_FoF1 = zeros((N, N))
                # for jj in range(N):
                #     for j in range(N):
                #         force_FoF1[jj, j] = -0.5 * Ecouple * sin(positions[jj] - positions[j])

                # energy_flow[ii, i] = trapz(trapz(-flux_array[1, ...] * force_FoF1)) * timescale
                energy_flow[ii, i] = -trapz(trapz(dflux_array[1, ...] * log(prob_ss_array))) * timescale

            # plot line at coupling strength corresponding to max power
            maxpos = argmax(output_power[:, i], axis=0)
            for j in range(4):
                axarr[j, i].axvline(Ecouple_array_tot[maxpos], linestyle='--', color=colorlst[k])

            axarr[0, i].plot(Ecouple_array_tot, flux[:, i], 'o', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[1, i].plot(Ecouple_array_tot, output_power[:, i], 'o', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[2, i].plot(Ecouple_array_tot, energy_flow[:, i], 'o', color=colorlst[k], label=labellst[k],
                             markersize=6)
            axarr[3, i].plot(Ecouple_array_tot, learning_rate[:, i], 'o', color=colorlst[k], label=labellst[k],
                             markersize=6)

            for j in range(4):
                axarr[j, i].yaxis.offsetText.set_fontsize(14)
                # axarr[j, i].ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
                axarr[j, i].tick_params(axis='y', labelsize=14)
                axarr[j, i].set_xscale('log')
                axarr[j, i].set_yscale('log')
                axarr[j, i].spines['right'].set_visible(False)
                axarr[j, i].spines['top'].set_visible(False)
                axarr[j, i].set_xlim((2, 150))
                # axarr[j, i].set_ylim(bottom=0)
            axarr[0, i].set_ylim((0.05, 30))
            axarr[1, i].set_ylim((0.05, 150))
            axarr[2, i].set_ylim((0.02, 10))
            axarr[3, i].set_ylim((0.02, 10))

            axarr[0, i].set_title(r'$%.0f$' % psi1_array[i], fontsize=18)

    # axarr[3, 2].set_ylim(top=6)
    axarr[0, 0].set_ylabel(r'$J_1 (\rm rad \cdot s^{-1})$', fontsize=14)
    axarr[1, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1})$', fontsize=14)
    axarr[2, 0].set_ylabel(r'$\beta \dot{E}_{\rm F_o \to F_1} (\rm s^{-1})$', fontsize=14)
    axarr[3, 0].set_ylabel(r'$\ell_{\rm F_o \to F_1} (\rm nats \cdot s^{-1})$', fontsize=14)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$\mu_{\rm H^+}$', ha='center', fontsize=20)
    f.text(0.5, 0.05, r'$E_{\rm couple}$', ha='center', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi))


def plot_super_grid_phi(target_dir, dt):  # grid of plots of output power, dissipation, MI, rel. entropy, learning rate
    Ecouple_array_tot = array([4.0, 8.0, 16.0])
    phase_array = array([0.0, 0.175, 0.349066, 0.524, 0.698132, 0.873, 1.0472, 1.222, 1.39626, 1.571, 1.74533, 1.92,
                         2.0944])
    phi_labels = [r'$0$', r'$2 \pi/3$', r'$4 \pi/3$', r'$2 \pi$']

    output_power = zeros((phase_array.size, Ecouple_array_tot.size))
    flux = zeros((phase_array.size, Ecouple_array_tot.size))
    energy_flow = zeros((phase_array.size, Ecouple_array_tot.size))
    learning_rate = zeros((phase_array.size, Ecouple_array_tot.size))

    output_file_name = (
            target_dir + "results/" + "Super_grid_phi_2_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}" + "_.pdf")

    f, axarr = plt.subplots(4, 3, sharex='all', figsize=(8, 8))

    for i, Ecouple in enumerate(Ecouple_array_tot):
        for ii, phi in enumerate(phase_array):
            # if Ecouple in Ecouple_array and phi in array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944]):
            #     input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/190624_Twopisweep_complete_set/" +
            #                        "flux_power_efficiency_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
            #                        "_outfile.dat")
            # else:
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/data/FP_Full_2D/191221_morepoints/" +
                               "flux_power_efficiency_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" +
                               "_outfile.dat")
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                                     usecols=(2, 4))
                flux_y = data_array[ii, 0]
                power_y = data_array[ii, 1]
            except OSError:
                print('Missing file power')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple))

            # grab output power
            output_power[ii, i] = -2 * pi * timescale * power_y
            # calculate probability current
            flux[ii, i] = 2 * pi * timescale * flux_y

        for ii, phi in enumerate(phase_array):
            if Ecouple in Ecouple_array and phi in array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533, 2.0944]):
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_Twopisweep_complete_set/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            else:
                input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
                                   "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                                   "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                    usecols=(0, 1, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))  # check grid size
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

            # calculate energy flow
            flux_array = zeros((2, N, N))
            calc_flux(prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)

            force_FoF1 = zeros((N, N))
            for jj in range(N):
                for j in range(N):
                    force_FoF1[jj, j] = -0.5 * Ecouple * sin(positions[jj] - positions[j])

            energy_flow[ii, i] = trapz(trapz(-flux_array[1, ...] * force_FoF1, dx=dx), dx=dx) * timescale

            # calculate learning rate
            dflux_array = empty((2, N, N))
            derivative_flux(flux_array, dflux_array, N)

            learning = dflux_array[1, ...] * log(prob_ss_array.sum(axis=0)/prob_ss_array)

            learning_rate[ii, i] = trapz(trapz(learning, dx=1, axis=1), dx=1) * timescale

        # plot line at coupling strength corresponding to max and min power
        maxpos = argmax(output_power[:, i], axis=0)
        minpos = argmin(output_power[:, i], axis=0)
        for j in range(4):
            axarr[j, i].axvline(phase_array[maxpos], linestyle='--', color='C1')
            axarr[j, i].axvline(phase_array[minpos], linestyle='--', color='C1')
        axarr[3, i].axhline(0, color='black')

        axarr[0, i].plot(phase_array, flux[:, i], 'o', markersize=6, color='C1')
        axarr[1, i].plot(phase_array, output_power[:, i], 'o', markersize=6, color='C1')
        axarr[2, i].plot(phase_array, energy_flow[:, i], 'o', markersize=6, color='C1')
        axarr[3, i].plot(phase_array, learning_rate[:, i], 'o', markersize=6, color='C1')

        for j in range(4):
            axarr[j, i].yaxis.offsetText.set_fontsize(14)
            axarr[j, i].ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
            axarr[j, i].tick_params(axis='y', labelsize=14)
            axarr[j, i].spines['right'].set_visible(False)
            axarr[j, i].spines['top'].set_visible(False)
            axarr[j, i].set_xticks([0, pi / 9, 2 * pi / 9, pi / 3, 4 * pi / 9, 5 * pi / 9, 2 * pi / 3])
            axarr[j, i].set_xticklabels(['$0$', '', '', '$1/2$', '', '', '$1$'])
            if j < 3 and i > 0:
                axarr[j, i].set_ylim(bottom=0)

        axarr[0, i].set_title(r'$%.0f$' % Ecouple_array_tot[i], fontsize=18)

    axarr[0, 0].set_ylabel(r'$J_1 (\rm rad \cdot s^{-1})$', fontsize=14)
    axarr[1, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1})$', fontsize=14)
    axarr[2, 0].set_ylabel(r'$\beta \dot{E}_{\rm F_o \to F_1} (\rm s^{-1})$', fontsize=14)
    axarr[3, 0].set_ylabel(r'$\ell_{\rm F_o \to F_1} (\rm nats \cdot s^{-1})$', fontsize=14)
    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$E_{\rm couple}$', ha='center', fontsize=20)
    f.text(0.5, 0.05, r'$n \phi (\rm rot^{-1})$', ha='center', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2))


def plot_nn_learning_rate_Ecouple(input_dir, dt):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x', 'p']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    phi = 0.0
    learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 6))
    axarr.axhline(0, color='black', label='_nolegend_')

    output_file_name = input_dir + "results/" + \
                       "Learning_rate_Ecouple_n0_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log_.pdf"

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = target_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                              "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                              "_outfile.dat"
            try:
                data_array = loadtxt(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))
                learning_rate[ii, j] = data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, 3.0, Ecouple))

        axarr.plot(Ecouple_array_tot, learning_rate[:, j],
                   color=color_lst[j], label=num_min, markersize=6, marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(16)
    axarr.tick_params(axis='both', labelsize=16)
    axarr.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr.set_ylabel(r'$\ell_1 \, (\rm nats/ s)$', fontsize=20)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')
    # axarr.set_yscale('log')

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_1$', fontsize=14,
                       loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(16)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi))


def plot_n0_learning_rate_Ecouple(input_dir, dt):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    phi = 0.0
    learning_rate = zeros((Ecouple_array_tot.size, min_array.size))

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 6))
    axarr.axhline(0, color='black', label='_nolegend_')

    output_file_name = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" + \
                       "Learning_rate_Ecouple_n0_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log_.pdf"

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = input_dir + "data/200915_energyflows/E0_{0}_E1_{1}/n1_{4}_n2_{5}/" + \
                              "power_heat_info_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + \
                              "_outfile.dat"
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_minima2, Ecouple))
                learning_rate[ii, j] = data_array[6]
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_minima2, Ecouple))

        axarr.plot(Ecouple_array_tot, learning_rate[:, j], color=color_lst[j], label=num_min, markersize=6,
                   marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(16)
    axarr.tick_params(axis='both', labelsize=16)
    axarr.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    axarr.set_ylabel(r'$\ell_1 (\rm nats \cdot s^{-1})$', fontsize=20)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xscale('log')

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14,
                       loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(16)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi))


def plot_nn_learning_rate_phi(input_dir, dt):  # plot power and efficiency as a function of the coupling strength
    markerlst = ['D', 's', 'o', 'v', 'x']
    color_lst = ['C2', 'C3', 'C1', 'C4', 'C6']
    Ecouple = 16.0
    phase_array_1 = array([0.0, 1.0472, 2.0944, 3.14159, 4.18879, 5.23599])
    phase_array_2 = array([0.0, 0.5236, 1.0472, 1.5708, 2.0944, 2.618])
    phase_array_3 = array([0.0, 0.349066, 0.698132, 1.0472, 1.39626, 1.74533])
    phase_array_6 = array([0.0, 0.1745, 0.349066, 0.5236, 0.698132, 0.8727])
    phase_array_12 = array([0.0, 0.08727, 0.17453, 0.2618, 0.34633, 0.4363])
    learning_rate = zeros((6, min_array.size))

    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 6))
    axarr.axhline(0, color='black', label='_nolegend_')

    output_file_name = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/results/" + \
                       "Learning_rate_Ecouple_n0_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_phi_{4}_log_.pdf"

    # Fokker-Planck results (barriers)
    for j, num_min in enumerate(min_array):
        if num_min == 1.0:
            phase_array = phase_array_1
        elif num_min == 2.0:
            phase_array = phase_array_2
        elif num_min == 3.0:
            phase_array = phase_array_3
        elif num_min == 6.0:
            phase_array = phase_array_6
        else:
            phase_array = phase_array_12

        for ii, phi in enumerate(phase_array):
            input_file_name = "/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190729_varying_n/" + "n1/" + \
                              "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" + \
                              "_outfile.dat"
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_min, num_min, phi),
                    usecols=(0, 3, 4, 5, 6, 7, 8))
                N = int(sqrt(len(data_array)))  # check grid size
                dx = 2 * math.pi / N
                positions = linspace(0, 2 * math.pi - dx, N)
                prob_ss_array = data_array[:, 0].T.reshape((N, N))
                drift_at_pos = data_array[:, 1:3].T.reshape((2, N, N))
                diffusion_at_pos = data_array[:, 3:].T.reshape((4, N, N))
            except OSError:
                print('Missing file')
                print(input_file_name.format(E0, E1, psi_1, psi_2, num_min, num_min, Ecouple))

            # calculate learning rate
            flux_array = zeros((2, N, N))
            calc_flux(positions, prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
            dflux_array = empty((2, N, N))
            derivative_flux(flux_array, dflux_array, N, dx)
            for i in range(N):
                for jj in range(N):
                    if prob_ss_array[i, jj] == 0:
                        prob_ss_array[i, jj] = 1e-18

            learning = dflux_array[1, ...] * log(prob_ss_array.sum(axis=0) / prob_ss_array)

            learning_rate[ii, j] = trapz(trapz(learning, dx=1, axis=1), dx=1) * timescale

    phase_array_1 = append(phase_array_1, 6.28319)
    print(learning_rate)
    for j, num_min in enumerate(min_array):
        axarr.plot(phase_array_1, append(learning_rate[:, j], learning_rate[0, j]), color=color_lst[j], label=num_min,
                   markersize=6, marker=markerlst[j], linestyle='-')

    # formatting
    axarr.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axarr.yaxis.offsetText.set_fontsize(16)
    axarr.tick_params(axis='both', labelsize=16)
    axarr.set_xlabel(r'$n \phi \ (\rm rad)$', fontsize=20)
    axarr.set_ylabel(r'$\ell_{\rm F_o \to F_1} (\rm nats \cdot s^{-1})$', fontsize=20)
    axarr.spines['right'].set_visible(False)
    axarr.spines['top'].set_visible(False)
    axarr.set_xticks([0, pi / 3, 2 * pi / 3, pi, 4 * pi / 3, 5 * pi / 3, 2 * pi])
    axarr.set_xticklabels(['$0$', '', '', '$\pi$', '', '', '$2 \pi$'])

    leg = axarr.legend(['$1$', '$2$', '$3$', '$6$', '$12$'], title=r'$n_{\rm o}$', fontsize=14,
                       loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(16)

    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, phi))


def compare_info(target_dir):
    phi = 0.0
    labels = ['nostalgia', '$\partial_1 J$', 'Conditional entropy', '$\partial_1 \log P$']
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    learning_rate = zeros((4, 2, Ecouple_array_tot.size))
    dt = 5e-2

    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/201016_dip/" +
                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                        "_outfile.dat")
    output_file_name = (
            target_dir + "results/" + "Learning_rate_split_dlogP_Ecouple_"
            + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_log.pdf")

    f, ax = plt.subplots(1, 1, sharex='all', sharey='none', figsize=(8, 6))

    for ii, Ecouple in enumerate(Ecouple_array_tot):
        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                usecols=(0, 1, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array)))
            dx = 2 * math.pi / N
            positions = linspace(0, 2 * math.pi - dx, N)
            prob_ss_array = data_array[:, 0].T.reshape((N, N))
            prob_eq_array = data_array[:, 1].T.reshape((N, N))
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

        # nostalgia
        # step_X = empty((N, N))
        # step_probability_X(step_X, prob_ss_array, drift_at_pos, diffusion_at_pos, N, dx, dt)
        #
        # mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
        # Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)
        #
        # pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
        # Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)

        # learning_rate[0, ii] = timescale * (Imem - Ipred) / dt

        # learning_rate[0, 0, ii] = timescale * (Imem) / dt
        # learning_rate[0, 1, ii] = timescale * (Ipred) / dt

        # leanring rate try 1
        # flux_array = empty((2, N, N))
        # calc_flux_2(positions, prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
        #
        # dflux_array = empty((2, N, N))
        # derivative_flux(flux_array, dflux_array, N, dx)
        #
        # learning = dflux_array[1, ...] * log(prob_ss_array.sum(axis=0) / prob_ss_array)
        #
        # learning_rate[1, ii] = trapz(trapz(learning)) * timescale
        #
        # # learning rate try 2
        # cond = dflux_array[1, ...] * (log(step_X / step_X.sum(axis=0)) + 1)
        # learning_rate[2, ii] = -trapz(trapz(cond)) * timescale
        #
        Hcouple = empty((N, N))
        # learning rate try 3
        for i in range(N):
            for j in range(N):
                if prob_ss_array[i, j] == 0:
                    prob_ss_array[i, j] = 10**(-18)

                Hcouple[i, j] = -0.5 * Ecouple * sin(positions[i] - positions[j])

        dPxy = empty((N, N))
        calc_derivative(prob_ss_array, dPxy, N, dx, 1)

        Py = prob_ss_array.sum(axis=0)
        Pxgy = prob_ss_array/Py
        dPxgy = empty((N, N))
        calc_derivative(Pxgy, dPxgy, N, dx, 1)

        # learning = prob_ss_array * (Hcouple + (dPxy/prob_ss_array)) * (dPxgy/Pxgy)
        #
        # learning_rate[3, ii] = -trapz(trapz(learning)) * timescale * 10**(-3)

        learning_rate[3, 0, ii] = -trapz(trapz(prob_ss_array * Hcouple * (dPxgy/Pxgy))) * timescale * 10 ** (-3)
        learning_rate[3, 1, ii] = -trapz(trapz(prob_ss_array * (dPxy/prob_ss_array) * (dPxgy/Pxgy))) * timescale * 10 ** (-3)

    # for i in range(4):
    ax.plot(Ecouple_array_tot, learning_rate[3, 0, :], '-o', markersize=8, label=r'$\ell^{\rm F}_1$')
    ax.plot(Ecouple_array_tot, learning_rate[3, 1, :], '-o', markersize=8, label=r'$\ell^{\rm B}_1$')
    ax.plot(Ecouple_array_tot, learning_rate[3, 1, :] + learning_rate[3, 0, :], '-o', markersize=8, label='$\ell_1$')

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim((-0, 3))
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    # ax.set_ylabel(r'$\rm \dot{E}_{\rm o \to 1}$', fontsize=20)
    ax.set_ylabel(r'$\dot{I}_1 (\rm nats/s)$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    leg = ax.legend(fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    # f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_entropy_production_Ecouple(target_dir):
    phase_shift = 0.0
    psi1_array = array([4.0])
    psi2_array = array([-2.0])
    Ecouple_array_tot = sort(concatenate((Ecouple_array, Ecouple_array_double)))
    gamma0 = 1000.0
    gamma1 = 100.0

    output_file_name = (target_dir + "results/" +
                        "Entropy_prod_gamma1_100_E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_phase_{6}" + ".pdf")

    for psi_1 in psi1_array:
        for psi_2 in psi2_array:
            # calculate entropy production
            integrate_entropy_X = empty(Ecouple_array_tot.size)
            integrate_entropy_Y = empty(Ecouple_array_tot.size)
            integrate_entropy_sum = empty(Ecouple_array_tot.size)
            integrate_entropy_diff = empty(Ecouple_array_tot.size)
            # integrate_heat_X = empty(Ecouple_array_tot.size)
            # integrate_heat_Y = empty(Ecouple_array_tot.size)
            # integrate_LR = empty(Ecouple_array_tot.size)
            # integrate_entropy2_X = empty(Ecouple_array_tot.size)
            # integrate_entropy2_Y = empty(Ecouple_array_tot.size)

            for ii, Ecouple in enumerate(Ecouple_array_tot):
                if Ecouple in Ecouple_array_tot:
                    input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/210329_friction/gamma1_100/" +
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
                    positions = linspace(0, 2 * math.pi - dx, N)
                    # print('Grid size: ', N)

                    prob_ss_array = data_array[:, 0].reshape((N, N))
                    potential_at_pos = data_array[:, 1].reshape((N, N))
                    drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
                    diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))

                    for i in range(N):
                        for j in range(N):
                            if prob_ss_array[i, j] == 0:
                                prob_ss_array[i, j] = 10e-18

                    flux_array = zeros((2, N, N))
                    calc_flux_2(positions, prob_ss_array, drift_at_pos, diffusion_at_pos, flux_array, N, dx)
                    flux_array = asarray(flux_array)

                    integrate_entropy_X[ii] = gamma0 * trapz(trapz(flux_array[0, ...]**2 / prob_ss_array)) * timescale
                    integrate_entropy_Y[ii] = gamma1 * trapz(trapz(flux_array[1, ...]**2 / prob_ss_array)) * timescale
                    # integrate_entropy_sum[ii] = trapz(trapz(
                    #     (av_gamma * flux_array[0, ...] + av_gamma * flux_array[1, ...]) ** 2 / prob_ss_array)
                    # ) * timescale
                    # integrate_entropy_diff[ii] = trapz(trapz(
                    #     (av_gamma * flux_array[0, ...] - av_gamma * flux_array[1, ...]) ** 2 / prob_ss_array)
                    # ) * timescale

                    # calculate heat flow
                    # dpotential_x = zeros((N, N))
                    # dpotential_y = zeros((N, N))
                    # calc_derivative(potential_at_pos, dpotential_x, N, dx, 0)
                    # calc_derivative(potential_at_pos, dpotential_y, N, dx, 1)
                    #
                    # integrate_heat_X[ii] = trapz(trapz(flux_array[0, ...] * (dpotential_x - psi_1))) * timescale
                    # integrate_heat_Y[ii] = trapz(trapz(flux_array[1, ...] * (dpotential_y - psi_2))) * timescale
                    #
                    # # calculate learning rate
                    # dflux_array = empty((2, N, N))
                    # derivative_flux(flux_array, dflux_array, N, dx)
                    #
                    # for i in range(N):
                    #     for j in range(N):
                    #         if prob_ss_array[i, j] == 0:
                    #             prob_ss_array[i, j] = 10e-18
                    #
                    # integrate_LR[ii] = -trapz(trapz(dflux_array[1, ...] * log(prob_ss_array))) * timescale

                except OSError:
                    print('Missing file')
                    print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))

            # integrate_entropy2_X = -integrate_heat_X + integrate_LR
            # integrate_entropy2_Y = -integrate_heat_Y - integrate_LR

            # plot entropy production
            plt.figure()
            f, ax = plt.subplots(1, 1)
            ax.plot(Ecouple_array_tot, integrate_entropy_X, '-o', label=r'$\dot{\Sigma}_{\rm o}$', color='tab:blue')
            ax.plot(Ecouple_array_tot, integrate_entropy_Y, '-v', label=r'$\dot{\Sigma}_1$', color='tab:blue')
            ax.plot(Ecouple_array_tot, integrate_entropy_Y + integrate_entropy_X, '-o', label=r'$\dot{\Sigma}$',
                    color='tab:orange')
            # ax.plot(Ecouple_array_tot, 0.5*integrate_entropy_sum, '-o', label=r'$\dot{\Sigma}_{\bar{\theta}}$',
            #         color='tab:green')
            # ax.plot(Ecouple_array_tot, 0.5*integrate_entropy_diff, '-v', label=r'$\dot{\Sigma}_{\Delta \theta}$',
            #         color='tab:green')
            # ax.plot(Ecouple_array_tot, integrate_entropy2_X, '--o', label=r'$\dot{\Sigma}_{\rm o}$', color='tab:orange')
            # ax.plot(Ecouple_array_tot, integrate_entropy2_Y, '--v', label=r'$\dot{\Sigma}_1$', color='tab:orange')
            # ax.set_ylim((3, 300))
            ax.set_xlim((2, None))

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=14)
            ax.set_ylabel(r'$\dot{\Sigma} \, (s^{-1})$', fontsize=14)
            # ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
            ax.tick_params(axis='both', labelsize=14)
            ax.yaxis.offsetText.set_fontsize(14)
            ax.legend(fontsize=12, frameon=False, ncol=1)

            f.tight_layout()
            f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phase_shift))


if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    # plot_ITQ_Ecouple(target_dir, 'learning_rate', 5e-2)  # options 'nostalgia', 'learning_rate', 'mutual_info',
    # 'relative_entropy' and the last option is dt.
    # plot_MI_Ecouple(target_dir, 5e-2)
    # dt = 0.001 is the standard used in the simulations.
    # plot_learning_rates_Ecouple(target_dir)
    # plot_nostalgia_Ecouple_grid(target_dir)
    # plot_correlation_nostalgia_power_peaks(target_dir)
    # plot_ITQ_phi(target_dir, 'nostalgia', 0.001)
    # plot_super_grid(target_dir, 5e-2)
    # plot_super_grid_peak(target_dir, 5e-2)
    # plot_super_grid_phi(target_dir, 5e-2)
    # plot_nn_learning_rate_Ecouple(target_dir, 5e-2)
    # plot_nn_learning_rate_phi(target_dir, 5e-2)
    # plot_n0_learning_rate_Ecouple(target_dir, 5e-2)
    # compare_info(target_dir)
    plot_entropy_production_Ecouple(target_dir)
