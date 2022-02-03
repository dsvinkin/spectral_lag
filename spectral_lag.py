import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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

def main():

    
    data = np.loadtxt('GRB20211227_T84726_64ms_BAT.thr')
    i_beg = 3737
    i_end = 3785
    n_shift = 8
    res = 0.064
    
    """
    data = np.loadtxt('GRB20211227_T84726_128ms_BAT.thr')
    i_beg = 1869
    i_end = 1892
    n_shift = 4
    res = 0.128
    """

    dic_chan = {'Ch21':(3,1), 'Ch32':(5,3), 'Ch31':(5,1)}
    chan = 'Ch21'

    i_ch_low_e, i_ch_hi_e = dic_chan[chan]
 
    data_c = data[i_beg:i_end+1,:]

    arr_dt, arr_ccf, arr_ccf_err =\
        correlate(data_c[:,i_ch_hi_e],  data_c[:,i_ch_hi_e+1], 
                  data_c[:,i_ch_low_e], data_c[:,i_ch_low_e+1], n_shift)

    n_sim = 200
    lst_ccf_sim, lst_ccf_sim_fit, lst_ccf_sim_fit_argmax = \
         make_sim_fits(data_c, i_ch_low_e, i_ch_hi_e, n_shift, n_sim)

    x = np.linspace(-0.5, 0.5, 20)
    for i in range(n_sim):
        plt.plot(arr_dt*res, lst_ccf_sim[i], c='k', alpha=0.1)
        y = poly(x, *lst_ccf_sim_fit[i])
        plt.plot(x, y, c='r', alpha=0.1)
        plt.axvline(lst_ccf_sim_fit_argmax[i], 0.0, 1.5, c='r', alpha=0.1)

    x_max = np.median(lst_ccf_sim_fit_argmax)
    err_dn = np.quantile(lst_ccf_sim_fit_argmax, 0.16) - x_max
    err_up = np.quantile(lst_ccf_sim_fit_argmax, 0.84) - x_max
    print("lag{:s}: {:8.3f} ({:+.3f}, {:+.3f})".format(chan, x_max, err_dn, err_up))

    plt.errorbar(arr_dt*res, arr_ccf, yerr=arr_ccf_err, label=f'{chan}')
    
    plt.legend()
    plt.grid()
    plt.xlim(-0.6,0.6)
    plt.ylim(0.4, 1.2)
    plt.show()

if __name__ == "__main__":
    main()