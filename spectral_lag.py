import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
plt.style.use('./dstyle1.mplstyle')

def correlate(arr_1, arr_1_err, arr_2, arr_2_err, i_shift_max):
    """
    Code is based on
    1) Link et al., 1993 https://ui.adsabs.harvard.edu/abs/1993ApJ...408L..81L/abstract
    2) Fenimore et al., 1995 https://ui.adsabs.harvard.edu/abs/1995ApJ...448L.101F/abstract

    """

    sig_1 = np.sum(arr_1**2 - arr_1_err**2)
    sig_2 = np.sum(arr_2**2 - arr_2_err**2)
    A_0 = (sig_1 * sig_2)**0.5

    #print(A_0)

    #err2_A_0 = np.sum( 4 * arr_1**3 + arr_1_err**2) * sig_2**2 + \
    #           np.sum( 4 * arr_2**3 + arr_2_err**2) * sig_1**2

    # use arr_[1|2]_err **2 instead of arr_[1|2] in error, remove secont term arr_[1|2]_err **2
    err2_A_0 = np.sum( 4 * arr_1**2 * arr_1_err**2) * sig_2**2 + \
               np.sum( 4 * arr_2**2 * arr_2_err**2) * sig_1**2

    err2_A_0 = (0.25 / A_0 ** 2) * err2_A_0

    #print(err2_A_0)

    arr_1 = np.pad(arr_1, (i_shift_max, i_shift_max+1), 'constant', constant_values=(0.0, 0.0))
    arr_1_err = np.pad(arr_1_err, (i_shift_max, i_shift_max+1), 'constant', 
        constant_values=(arr_1_err[0], arr_1_err[-1]))

    arr_ccf = np.zeros(2*i_shift_max+1)
    arr_sig2_cicj = np.zeros(2*i_shift_max+1)
    arr_dt = np.zeros(2*i_shift_max+1)  

    for i_shift in range(-i_shift_max, i_shift_max+1):
        
        fsum = 0.0
        fsum_cicj = 0.0
        for i in range(len(arr_2)):
            fsum += arr_1[i+i_shift+i_shift_max] * arr_2[i]

            #fsum_cicj += np.abs(arr_1[i+i_shift+i_shift_max]) * arr_2[i]**2 + \
            #    arr_1[i+i_shift+i_shift_max]**2 * np.abs(arr_2[i])

            fsum_cicj += arr_1_err[i+i_shift+i_shift_max] **2 * arr_2[i]**2 + \
                arr_1[i+i_shift+i_shift_max]**2 * arr_2_err[i]**2

        arr_ccf[i_shift+i_shift_max] = fsum 
        arr_sig2_cicj[i_shift+i_shift_max] = fsum_cicj
        arr_dt[i_shift+i_shift_max] = i_shift

    arr_ccf = arr_ccf / A_0
    arr_ccf_err = (arr_sig2_cicj / A_0 ** 2 + arr_ccf ** 2 / A_0 ** 2 * err2_A_0)**0.5

    #print(arr_sig2_cicj / A_0 ** 2)
    #print(arr_ccf ** 2 / A_0 ** 2)

    return arr_dt, arr_ccf, arr_ccf_err

def poly(x, a0, a1, a2):
    return a0 + a1 * x + a2 * x**2

def fit_ccf(arr_dt, arr_ccf, arr_ccf_err):

    p_init = [1.0, 0.0, -2.0]
    popt, pcov = curve_fit(poly, arr_dt, arr_ccf, sigma=arr_ccf_err, p0=p_init)
    #print(popt)
    #print(pcov)
    return popt, pcov, -popt[1]/2/popt[2]

def sim_data(data):

    data_sim = np.copy(data)
    for i in range(data_sim.shape[0]):
        for j in [1,3,5]:
            data_sim[i,j] = np.random.normal(loc=data[i,j], scale=data[i,j+1])

    return data_sim

def make_sim_fits(data, i_ch_low_e, i_ch_hi_e, n_shift, n_sim):

    lst_ccf_sim = []
    lst_ccf_sim_fit = []
    lst_ccf_sim_fit_argmax = []
    
    res = data[1,0] - data[0,0]

    for i_sim in range(n_sim):
        data_sim = sim_data(data)
        arr_dt, arr_ccf, arr_ccf_err =\
            correlate(data_sim[:,i_ch_hi_e], data_sim[:,i_ch_hi_e+1], 
                      data_sim[:,i_ch_low_e], data_sim[:,i_ch_low_e+1], n_shift)

        popt, pcov, x_max = fit_ccf(arr_dt*res, arr_ccf, arr_ccf_err)

        lst_ccf_sim_fit.append(popt)
        lst_ccf_sim_fit_argmax.append(x_max)
        lst_ccf_sim.append(arr_ccf)

    return lst_ccf_sim, lst_ccf_sim_fit, lst_ccf_sim_fit_argmax


def set_layout(fig):

    # Set panel parameters
    left = 0.10
    top = 0.95
    width = 0.8
    heigt_ax = 0.28
    
    
    # rect [left, bottom, width, height] 
    
    lst_rect = []
    lst_rect.append([left, top - heigt_ax, width, heigt_ax])
    lst_rect.append([left, top - 2*heigt_ax, width, heigt_ax])
    lst_rect.append([left, top - 3*heigt_ax, width, heigt_ax])

    lst_axis = []
    for i in range(len(lst_rect)):
        if i > 0:
            lst_axis.append(fig.add_axes(lst_rect[i], sharex=lst_axis[0]))
        else:
            lst_axis.append(fig.add_axes(lst_rect[i]))

        if i < len(lst_rect) - 1:
            lst_axis[-1].tick_params(labelbottom=False)

    return lst_axis

def plot_lags(ax, str_label, arr_dt, arr_ccf, arr_ccf_err, lst_ccf_sim, lst_ccf_sim_fit, lst_ccf_sim_fit_argmax):
    
    n_sim = len(lst_ccf_sim)
    res = 0.064

    x = np.linspace(-0.5, 0.5, 20)
    """
    for i in range(n_sim):
        ax.plot(arr_dt*res, lst_ccf_sim[i], c='k', alpha=0.1)
        y = poly(x, *lst_ccf_sim_fit[i])
        ax.plot(x, y, c='r', alpha=0.1)
        ax.axvline(lst_ccf_sim_fit_argmax[i], 0.0, 1.5, c='r', alpha=0.1)
    """
    ax.errorbar(arr_dt*res, arr_ccf, yerr=arr_ccf_err, fmt='o', c='k', ) # label=str_label
    popt, pcov, x_max = fit_ccf(arr_dt*res, arr_ccf, arr_ccf_err)
    y = poly(x, *popt)
    ax.plot(x, y, c='k', alpha=0.5)

    x_max, err_dn, err_up = get_conf_int(lst_ccf_sim_fit_argmax, 0.68)
    ax.axvline(x_max, 0.0, 1.5, ls='--', c='k', alpha=0.8)
    ax.axvspan(x_max + err_dn, x_max + err_up, color='k', alpha=0.25)

    #leg = ax.legend(loc='lower right', fancybox=False)
    #for item in leg.legendHandles:
    #    item.set_visible(False)

    ax.text(-0.55, 0.45, str_label)

    ax.grid()
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.4, 1.19)

def get_conf_int(arr_x, cf):
    """Calculate confidence interval 

    Args:
        arr_x (np.array): random variable realizations
        cf (float): confidence level

    Returns:
        [tuple]: median of arr_x, lower and upper at conf. lebvel ci
    """   

    x_ = np.median(arr_x)
    err_dn = np.quantile(arr_x, (1-cf)/2) - x_
    err_up = np.quantile(arr_x, (1+cf)/2) - x_

    return x_, err_dn, err_up

def main():

    
    data = np.loadtxt('GRB20211227_T84726_64ms_BAT.thr')
    i_beg = 3737
    i_end = 3785
    n_shift = 8
    res = 0.064

    n_sim = 1000
    
    """
    data = np.loadtxt('GRB20211227_T84726_128ms_BAT.thr')
    i_beg = 1869
    i_end = 1892
    n_shift = 4
    res = 0.128
    """

    dic_chan = {'Ch21':(3,1), 'Ch32':(5,3), 'Ch31':(5,1)}
    dic_chan_dec = {'Ch21':'25-50 keV - 15-25 keV', 
        'Ch32':'50-100 keV - 25-50 keV', 
        'Ch31':'50-100 keV - 15-25 keV'}
    lst_chan = 'Ch21 Ch32 Ch31'.split()

    data_c = data[i_beg:i_end+1,:]
    
    fig = plt.figure(figsize=(8,8))

    lst_axis = set_layout(fig)

    for i, chan in enumerate(lst_chan):

        i_ch_low_e, i_ch_hi_e = dic_chan[chan]
 
        arr_dt, arr_ccf, arr_ccf_err =\
            correlate(data_c[:,i_ch_hi_e],  data_c[:,i_ch_hi_e+1], 
                    data_c[:,i_ch_low_e], data_c[:,i_ch_low_e+1], n_shift)

    
        lst_ccf_sim, lst_ccf_sim_fit, lst_ccf_sim_fit_argmax = \
                make_sim_fits(data_c, i_ch_low_e, i_ch_hi_e, n_shift, n_sim)

    
        plot_lags(lst_axis[i], dic_chan_dec[chan], arr_dt, arr_ccf, arr_ccf_err, lst_ccf_sim, lst_ccf_sim_fit, lst_ccf_sim_fit_argmax)

        x_max, err_dn, err_up = get_conf_int(lst_ccf_sim_fit_argmax, 0.68)
        print("lag{:s}: {:8.3f} ({:+.3f}, {:+.3f})".format(chan, x_max, err_dn, err_up))

    lst_axis[2].set_xlabel("t$_\mathrm{lag}$ (s)")
    lst_axis[0].set_ylim(0.4, 1.2)
    lst_axis[1].set_ylabel("CCF")
    fig.savefig('bat_lags.pdf')

if __name__ == "__main__":
    main()