import numpy as np
import autograd.numpy as ag_np


def calc_flow_into_symptomatic_T__forloop(
        f_T, proba_symp, pmf_dur_before_symp_J):
    ''' Compute flow into symptomatic state over T days using forloop method

    Args
    ----
    f_T : 1D array, shape (T,)
        Entry at index t counts the Flow into the infected state at time t
    proba_symp : float, positive
        Probability that an infected person will become symptomatic
    pmf_dur_before_symp_J : 1D array, shape (max_duration_in_days,) = (J,)
        Entry j gives probability that an infected persom will wait j days
        before becoming symptomatic, for j in 0, 1, .... J-1

    Returns
    -------
    g_T : 1D array, shape (T,)
        Entry at index t counts the flow into the symptomatic state at time t 
     
    Examples
    --------
    >>> proba_symp = 0.5
    >>> pmf_dur_before_symp = np.asarray([0.1, 0.5, 0.3, 0.1])
    >>> f_T = 10 * np.arange(1, 11)
    >>> calc_flow_into_symptomatic_T__convolution(
    ...     f_T, proba_symp, pmf_dur_before_symp)
    array([ 0.5,  3.5,  8. , 13. , 18. , 23. , 28. , 33. , 38. , 43. ])
    '''
    T = f_T.size
    J = pmf_dur_before_symp_J.size
    g_T = ag_np.zeros(T, dtype=np.float64)
    for t in range(T):
        # Size of look-back window at day t
        # Will be size 1 on day 0, 2 on day 1, ... up to J on day J-1 and on
        W = 1 + ag_np.minimum(t, J-1)
        assert W <= J
        # Indices of the look-back window at day t
        inds_W = ag_np.arange(t, t - W, -1)
        # Compute the flow on day t by summing over look-back window
        g_T[t] = proba_symp * (
            ag_np.inner(f_T[inds_W], pmf_dur_before_symp_J[:W]))
    return g_T

def calc_flow_into_symptomatic_T__convolution(
        f_T, proba_symp, pmf_dur_before_symp_J):
    ''' Compute flow into symptomatic state over T days using conv method

    Args
    ----
    f_T : 1D array, shape (T,)
        Entry at index t counts the Flow into the infected state at time t
    proba_symp : float, positive
        Probability that an infected person will become symptomatic
    pmf_dur_before_symp_J : 1D array, shape (max_duration_in_days,) = (J,)
        Entry j gives probability that an infected persom will wait j days
        before becoming symptomatic, for j in 0, 1, .... J-1

    Returns
    -------
    g_T : 1D array, shape (T,)
        Entry at index t counts the flow into the symptomatic state at time t 

    Examples
    --------
    >>> proba_symp = 0.5
    >>> pmf_dur_before_symp = np.asarray([0.1, 0.5, 0.3, 0.1])
    >>> f_T = 10 * np.arange(1, 11)
    >>> calc_flow_into_symptomatic_T__convolution(
    ...     f_T, proba_symp, pmf_dur_before_symp)
    array([ 0.5,  3.5,  8. , 13. , 18. , 23. , 28. , 33. , 38. , 43. ])
    '''
    T = f_T.size
    g_T = proba_symp * ag_np.convolve(f_T, pmf_dur_before_symp_J, mode='full')
    # 'Full' mode computes a few extra entries, we keep only first T
    return g_T[:T]

if __name__ == '__main__':
    print("hi")
    proba_symp = 0.5
    pmf_dur_before_symp = np.asarray([0.1, 0.5, 0.3, 0.1])
    
    # Define flow into infected
    f_T = 10 * np.arange(1, 11)

    # Compute flow into symptomatic in two ways
    for calc_flow_into_symptomatic_T in [
            calc_flow_into_symptomatic_T__forloop,
            calc_flow_into_symptomatic_T__convolution]:
        # Calculate the flow
        g_T = calc_flow_into_symptomatic_T(
            f_T, proba_symp, pmf_dur_before_symp)
        # Pretty print result
        print("Using method: %s" % calc_flow_into_symptomatic_T.__name__)
        print("   g_T.shape %s" % g_T.shape)
        print("   g_T value:")
        print(g_T)
        print("\n")
