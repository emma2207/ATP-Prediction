from numpy import array, linspace, loadtxt, append, pi, empty, sqrt, zeros, asarray, trapz, log, argmax
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from utilities import step_probability_X, calc_flux, calc_derivative_pxgy

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

N = 540  # N x N grid is used for Fokker-Planck simulations
dx = 2 * math.pi / N  # spacing between gridpoints
positions = linspace(0, 2 * math.pi - dx, N)  # gridpoints
timescale = 1.5 * 10**4  # conversion factor between simulation and experimental timescale

E0 = 2.0  # barrier height Fo
E1 = 2.0  # barrier height F1
psi_1 = 8.0  # chemical driving force on Fo
psi_2 = -4.0  # chemical driving force on F1
num_minima1 = 3.0  # number of barriers in Fo's landscape
num_minima2 = 3.0  # number of barriers in F1's landscape

Ecouple_array = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])  # coupling strengths
min_array = array([1.0, 2.0, 3.0, 6.0, 12.0])  # number of energy minima/ barriers


def plot_ITQ_Ecouple(target_dir, quantity, dt):  # grid of plots of the flux as a function of the phase offset
    # Ecouple_array_tot = array(
    #     [2.0, 2.83, 4.0, 5.66, 8.0, 10.0, 11.31, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 32.0,
    #      45.25, 64.0, 90.51, 128.0])
    # Ecouple_array_tot = array(
    #     [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.62, 32.0, 45.25, 64.0, 90.51, 128.0])
    Ecouple_array_tot = array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])

    if quantity == 'nostalgia':
        output_file_name = (
                target_dir + "results/" + "Nostalgia_Ecouple_"
                + "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n0_{4}_n1_{5}_phi_{6}" + "_.pdf")
    elif quantity == 'learning_rate':
        output_file_name = (
                target_dir + "results/" + "LearningRate_Ecouple_"
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

    # Fokker-Planck zero-barriers
    # phi = 0.0
    # information = zeros(Ecouple_array.size)
    #
    # for ii, Ecouple in enumerate(Ecouple_array):
    #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Zero-barriers-FP/2019-05-14/" +
    #                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
    #                        "_outfile.dat")
    #
    #     try:
    #         data_array = loadtxt(
    #             input_file_name.format(0.0, Ecouple, 0.0, psi_1, psi_2, num_minima1, num_minima2, phi),
    #             usecols=(0, 1, 3, 4, 5, 6, 7, 8))
    #         N = int(sqrt(len(data_array)))  # check grid size
    #         prob_ss_array = data_array[:, 0].T.reshape((N, N))
    #         prob_eq_array = data_array[:, 1].T.reshape((N, N))
    #         drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
    #         diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
    #     except OSError:
    #         print('Missing file')
    #         print(input_file_name.format(0.0, Ecouple, 0.0, psi_1, psi_2, num_minima1, num_minima2, phi))
    #
    #     if quantity == 'nostalgia':
    #         step_X = empty((N, N))
    #         dx = 2 * math.pi / N  # spacing between gridpoints
    #         step_probability_X(
    #             step_X, prob_ss_array, drift_at_pos, diffusion_at_pos,
    #             N, dx, dt
    #         )
    #
    #         # instantaneous memory
    #         mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
    #         Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)
    #
    #         # instantaneous predictive power
    #         pred_denom = ((step_X.sum(axis=1))[:, None] * (step_X.sum(axis=0))[None, :])
    #         Ipred = (step_X * log(step_X / pred_denom)).sum(axis=None)
    #
    #         information[ii] = timescale*(Imem - Ipred)/dt
    #
    #     elif quantity == 'learning_rate':
    #         flux_array = empty((2, N, N))
    #         calc_flux(
    #             positions, prob_ss_array, drift_at_pos, diffusion_at_pos,
    #             flux_array, N, dx
    #         )
    #
    #         Dpxgy = empty((N, N))
    #         calc_derivative_pxgy(
    #             prob_ss_array, prob_ss_array.sum(axis=0),
    #             Dpxgy,
    #             N, dx
    #         )
    #
    #         learning = flux_array[1, ...] * log(Dpxgy)
    #
    #         information[ii] = trapz(
    #             trapz(learning, dx=dx, axis=1), dx=dx
    #         )
    #
    #     elif quantity == 'mutual_info':
    #         # instantaneous memory
    #         mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
    #         Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)
    #
    #         information[ii] = Imem
    #
    #     elif quantity == 'relative_entropy':
    #         information[ii] = (prob_ss_array * log(prob_ss_array/prob_eq_array)).sum(axis=None)
    #
    # ax.plot(Ecouple_array, information, 'o', color='C0', label='$0$', markersize=8)

    # Fokker-Planck results (barriers)
    phi = 0.0
    information = zeros(Ecouple_array_tot.size)

    for ii, Ecouple in enumerate(Ecouple_array_tot):
        input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/200427_strongforces/" +
                           "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
                           "_outfile.dat")

        # if Ecouple in Ecouple_array:
        #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190624_phaseoffset/" +
        #                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
        #                        "_outfile.dat")
        # elif Ecouple in array([10.0, 12.0, 14.0, 18.0, 20.0, 22.0, 24.0]):
        #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/190610_phaseoffset_extra/" +
        #                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
        #                        "_outfile.dat")
        # else:
        #     input_file_name = ("/Users/Emma/Documents/Data/ATPsynthase/Full-2D-FP/191221_morepoints/" +
        #                        "reference_E0_{0}_Ecouple_{1}_E1_{2}_psi1_{3}_psi2_{4}_n1_{5}_n2_{6}_phase_{7}" +
        #                        "_outfile.dat")

        try:
            data_array = loadtxt(
                input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi),
                usecols=(0, 1, 3, 4, 5, 6, 7, 8))
            N = int(sqrt(len(data_array)))  # check grid size
            # print(N)
            prob_ss_array = data_array[:, 0].T.reshape((N, N))
            prob_eq_array = data_array[:, 1].T.reshape((N, N))
            drift_at_pos = data_array[:, 2:4].T.reshape((2, N, N))
            diffusion_at_pos = data_array[:, 4:].T.reshape((4, N, N))
        except OSError:
            print('Missing file')
            print(input_file_name.format(E0, Ecouple, E1, psi_1, psi_2, num_minima1, num_minima2, phi))

        if quantity == 'nostalgia':
            step_X = empty((N, N))
            dx = 2 * math.pi / N  # spacing between gridpoints
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

            information[ii] = timescale*(Imem - Ipred)/dt
        elif quantity == 'learning_rate':
            flux_array = empty((2, N, N))
            calc_flux(
                positions, prob_ss_array, drift_at_pos, diffusion_at_pos,
                flux_array, N, dx
            )

            Dpxgy = empty((N, N))
            calc_derivative_pxgy(
                prob_ss_array, prob_ss_array.sum(axis=0),
                Dpxgy,
                N, dx
            )

            learning = flux_array[1, ...] * Dpxgy

            information[ii] = trapz(
                trapz(learning, dx=dx, axis=1), dx=dx
            )

        elif quantity == 'mutual_info':
            mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
            Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

            information[ii] = Imem

        elif quantity == 'relative_entropy':
            information[ii] = (prob_ss_array * log(prob_ss_array/prob_eq_array)).sum(axis=None)

    # maxpos = argmax(information)
    # ax.axvline(Ecouple_array_tot[maxpos], linestyle='--', color='grey')
    ax.plot(Ecouple_array_tot, information, 'o', color='C1', label='$2$', markersize=8)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(14)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel(r'$\beta E_{\rm couple}$', fontsize=20)
    if quantity == 'nostalgia':
        ax.set_ylabel(r'$\ell_{\rm F_1} (\rm nats/s)$', fontsize=20)
    elif quantity == 'learning_rate':
        ax.set_ylabel(r'$\ell_1$', fontsize=20)
    elif quantity == 'mutual_info':
        ax.set_ylabel(r'$I(\theta_{\rm o}(t), \theta_1(t))$', fontsize=20)
    elif quantity == 'relative_entropy':
        ax.set_ylabel(r'$\mathcal{D}_{\rm KL}( P_{\rm ss} || P_{\rm eq} )$', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.set_ylim((0, None))

    leg = ax.legend(title=r'$\beta E_{\rm o} = \beta E_1$', fontsize=16, loc='best', frameon=False)
    leg_title = leg.get_title()
    leg_title.set_fontsize(20)

    # f.subplots_adjust(hspace=0.01)
    f.tight_layout()
    f.savefig(output_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, phi))


def plot_nostalgia_Ecouple_grid(target_dir, quantity):  # grid of plots of the flux as a function of the phase offset
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
    psi1_array = array([2.0, 4.0, 8.0])
    psi_ratio = array([8, 4, 2])
    phi = 0.0
    information = zeros((Ecouple_array_tot.size, psi1_array.size, psi_ratio.size))

    if quantity == 'nostalgia':
        output_file_name = (
                target_dir + "results/" + "Nostalgia_Ecouple_grid_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_.pdf")
    elif quantity == 'learning_rate':
        output_file_name = (
                target_dir + "results/" + "LearningRate_Ecouple_grid_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" +
                "_.pdf")

    f, axarr = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(8, 6))

    for i, psi_1 in enumerate(psi1_array):
        for j, ratio in enumerate(psi_ratio):
            psi_2 = -psi_1 / ratio

            # Fokker-Planck results (barriers)
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

                if quantity == 'nostalgia':
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
                elif quantity == 'learning_rate':
                    flux_array = empty((2, N, N))
                    calc_flux(
                        positions, prob_ss_array, drift_at_pos, diffusion_at_pos,
                        flux_array, N, dx
                    )

                    Dpxgy = empty((N, N))
                    calc_derivative_pxgy(
                        prob_ss_array, prob_ss_array.sum(axis=0),
                        Dpxgy,
                        N, dx
                    )

                    learning = flux_array[1, ...] * log(Dpxgy)
                    information[ii] = trapz(trapz(learning, dx=dx, axis=1), dx=dx)

            # maxpos = argmax(information, axis=0)
            # axarr[i, j].axvline(Ecouple_array_tot[maxpos[i, j]], linestyle='--', color='grey')
            axarr[i, j].plot(Ecouple_array_tot, information[:, i, j], 'o', color='C1', label='$2$', markersize=8)

            axarr[i, j].yaxis.offsetText.set_fontsize(14)
            axarr[i, j].tick_params(axis='y', labelsize=14)
            axarr[i, j].set_xscale('log')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            # axarr[i, j].set_ylim((0, 2.1*10**-7))

            if j == 0 and i > 0:
                axarr[i, j].yaxis.offsetText.set_fontsize(0)
            else:
                axarr[i, j].yaxis.offsetText.set_fontsize(14)

            if j == psi1_array.size - 1:
                axarr[i, j].set_ylabel(r'$%.0f$' % psi_ratio[::-1][i], labelpad=16, rotation=270, fontsize=18)
                axarr[i, j].yaxis.set_label_position('right')

            if i == 0:
                axarr[i, j].set_title(r'$%.0f$' % psi1_array[::-1][j], fontsize=18)

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.1, hspace=0.1)
    f.text(0.5, 0.01, r'$\beta E_{\rm couple}$', ha='center', fontsize=24)
    if quantity == 'nostalgia':
        f.text(0.01, 0.5, r'$I_{\rm mem} - I_{\rm pred}$', va='center', rotation='vertical',
               fontsize=24)
    elif quantity == 'learning_rate':
        f.text(0.01, 0.5, r'$\ell_1$', va='center', rotation='vertical',
               fontsize=24)
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
    Ecouple_array_tot = array(
        [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0, 22.63, 32.0, 45.25, 64.0, 90.51, 128.0])
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

    output_file_name = (
            target_dir + "results/" + "Super_grid_double_" + "E0_{0}_E1_{1}_n0_{2}_n1_{3}_phi_{4}" + "_.pdf")

    f, axarr = plt.subplots(5, 3, sharex='all', figsize=(8, 10))

    # Barrier-less data
    for i, psi_1 in enumerate(psi1_array):
        psi_2 = psi2_array[i]

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/" +
                               "fokker_planck/working_directory_cython/" + "plotting_data/"
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

        # plot output power
        axarr[0, i].plot(Ecouple_array2, power_y, '-', color='C0', label='$0$', linewidth=2)

        # plot dissipation
        axarr[1, i].plot(Ecouple_array2, dissipation_zero, '-', color='C0', label='$0$', linewidth=2)

        # plot mutual information
        axarr[2, i].plot(Ecouple_array, mutual_info_zero[:, i], 'o', color='C0', label='$0$', markersize=8)

        # plot learning rate
        axarr[3, i].plot(Ecouple_array, learning_rate_zero[:, i], 'o', color='C0', label='$0$', markersize=8)

        # plot relative entropy
        axarr[4, i].plot(Ecouple_array, rel_entropy_zero[:, i], 'o', color='C0', label='$0$', markersize=8)

    # Barrier data
    for i, psi_1 in enumerate(psi1_array):
        psi_2 = psi2_array[i]

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            input_file_name = ("/Users/Emma/sfuvault/SivakGroup/Emma/ATPsynthase/FokkerPlanck_2D_full/prediction/" +
                               "fokker_planck/working_directory_cython/" + "plotting_data/" + "flux_power_efficiency_" +
                               "E0_{0}_E1_{1}_psi1_{2}_psi2_{3}_n1_{4}_n2_{5}_Ecouple_{6}" + "_outfile.dat")
            try:
                data_array = loadtxt(
                    input_file_name.format(E0, E1, psi_1, psi_2, num_minima1, num_minima2, Ecouple),
                    usecols=(3, 4))
                if psi_1 == 4.0 and psi_2 == -2.0 and Ecouple in Ecouple_array:
                    power_x = data_array[0, 0]
                    power_y = data_array[0, 1]
                else:
                    power_x = data_array[0]
                    power_y = data_array[1]
            except OSError:
                print('Missing file power')

            # grab output power
            output_power[ii, i] = -2 * pi * timescale * power_y
            # calculate dissipation
            dissipation[ii, i] = 2 * pi * timescale * (power_x + power_y)

        for ii, Ecouple in enumerate(Ecouple_array_tot):
            if Ecouple in Ecouple_array:
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

            # calculate mutual information
            mem_denom = ((prob_ss_array.sum(axis=1))[:, None] * (prob_ss_array.sum(axis=0))[None, :])
            Imem = (prob_ss_array * log(prob_ss_array / mem_denom)).sum(axis=None)

            mutual_info[ii, i] = Imem

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

            learning_rate[ii, i] = timescale*(Imem - Ipred)/dt

            # calculate relative entropy
            rel_entropy[ii, i] = (prob_ss_array * log(prob_ss_array / prob_eq_array)).sum(axis=None)

        # plot line at coupling strength corresponding to max power
        maxpos = argmax(output_power[:, i], axis=0)
        for j in range(5):
            axarr[j, i].axvline(Ecouple_array_tot[maxpos], linestyle='--', color='grey')

        # plot output power
        axarr[0, i].plot(Ecouple_array_tot, output_power[:, i], 'o', color='C1', label='$2$', markersize=6)
        axarr[0, 0].set_ylabel(r'$\beta \mathcal{P}_{\rm ATP} (\rm s^{-1})$', fontsize=14)

        # plot dissipation
        axarr[1, i].plot(Ecouple_array_tot, dissipation[:, i], 'o', color='C1', label='$2$', markersize=6)
        axarr[1, 0].set_ylabel(r'$\beta (\mathcal{P}_{\rm H^+} -\mathcal{P}_{\rm ATP}) (\rm s^{-1}) $', fontsize=14)

        # plot mutual information
        axarr[2, i].plot(Ecouple_array_tot, mutual_info[:, i], 'o', color='C1', label='$2$', markersize=6)
        axarr[2, 0].set_ylabel(r'$I(\theta_{\rm o}(t), \theta_1(t)) (\rm nats)$', fontsize=14)

        # plot learning rate
        axarr[3, i].plot(Ecouple_array_tot, learning_rate[:, i], 'o', color='C1', label='$2$', markersize=6)
        axarr[3, 0].set_ylabel(r'$\ell_{\rm F_1} (\rm nats/s)$', fontsize=14)

        # plot relative entropy
        axarr[4, i].plot(Ecouple_array_tot, rel_entropy[:, i], 'o', color='C1', label='$2$', markersize=6)
        axarr[4, 0].set_ylabel(r'$\mathcal{D}_{\rm KL} ( P_{\rm ss} || P_{\rm eq} ) (\rm nats)$', fontsize=14)

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

    f.tight_layout()
    f.subplots_adjust(bottom=0.12, left=0.12, right=0.9, top=0.88, wspace=0.25, hspace=0.3)

    f.text(0.5, 0.95, r'$\mu_{\rm H^+}$', ha='center', fontsize=20)
    f.text(0.5, 0.05, r'$E_{\rm couple}$', ha='center', fontsize=20)
    f.savefig(output_file_name.format(E0, E1, num_minima1, num_minima2, phi))

if __name__ == "__main__":
    target_dir = "/Users/Emma/sfuvault/SivakGroup/Emma/ATP-Prediction/"
    plot_ITQ_Ecouple(target_dir, 'nostalgia', 5e-2)  # options 'nostalgia', 'learning_rate', 'mutual_info',
    # 'relative_entropy' and the last option is dt.
    # dt = 0.001 is the standard used in the simulations.
    # plot_nostalgia_Ecouple_grid(target_dir, 'learning_rate')  # options 'nostalgia', 'learning_rate'
    # plot_correlation_nostalgia_power_peaks(target_dir)
    # plot_ITQ_phi(target_dir, 'nostalgia', 0.001)
    # plot_super_grid(target_dir, 0.001)
