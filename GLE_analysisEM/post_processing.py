"""
Somes utilities function
"""
import numpy as np
import scipy.linalg
import numpy.polynomial.polynomial as poly
import scipy.integrate


def memory_kernel(ntimes, dt, coeffs, dim_x, noDirac=False):
    """
    Return the value of the estimated memory kernel

    Parameters
    ----------
    ntimes,dt: Number of timestep and timestep
    coeffs : Coefficients for diffusion and friction
    dim_x: Dimension of visible variables
    noDirac: Remove the dirac at time zero

    Returns
    -------
    timespan : array-like, shape (n_samples, )
        Array of time to evaluate memory kernel

    kernel_evaluated : array-like, shape (n_samples, dim_x,dim_x)
        Array of values of the kernel at time provided
    """
    Avv = coeffs["A"][:dim_x, :dim_x]
    Ahv = coeffs["A"][dim_x:, :dim_x]
    Avh = coeffs["A"][:dim_x, dim_x:]
    Ahh = coeffs["A"][dim_x:, dim_x:]
    Kernel = np.zeros((ntimes, dim_x, dim_x))
    for n in np.arange(ntimes):
        Kernel[n, :, :] = -np.matmul(Avh, np.matmul(scipy.linalg.expm(-1 * n * dt * Ahh), Ahv))
    if not noDirac:
        Kernel[0, :, :] = Kernel[0, :, :] + 2 * Avv
    return dt * np.arange(ntimes), Kernel


def memory_kernel_logspace(dt, coeffs, dim_x, noDirac=False):
    """
    Return the value of the estimated memory kernel

    Parameters
    ----------
    dt: Timestep
    coeffs : Coefficients for diffusion and friction
    dim_x: Dimension of visible variables
    noDirac: Remove the dirac at time zero

    Returns
    -------
    timespan : array-like, shape (n_samples, )
        Array of time to evaluate memory kernel

    kernel_evaluated : array-like, shape (n_samples, dim_x,dim_x)
        Array of values of the kernel at time provided
    """
    Avv = coeffs["A"][:dim_x, :dim_x]
    Ahv = coeffs["A"][dim_x:, :dim_x]
    Avh = coeffs["A"][:dim_x, dim_x:]
    Ahh = coeffs["A"][dim_x:, dim_x:]
    eigs = np.linalg.eigvals(Ahh)
    Kernel = np.zeros((150, dim_x, dim_x))
    final_time = 25 / np.min(np.abs(np.real(eigs)))
    times = np.logspace(np.log10(dt), np.log10(final_time), num=150)
    for n, t in enumerate(times):
        Kernel[n, :, :] = -np.matmul(Avh, np.matmul(scipy.linalg.expm(-1 * t * Ahh), Ahv))
    if not noDirac:
        Kernel[0, :, :] = Kernel[0, :, :] + 2 * Avv
    return times, Kernel


def memory_timescales(coeffs, dim_x):
    """
    Compute the eigenvalues of A_hh to get the timescale of the memory
    """
    return np.linalg.eigvals(coeffs["A"][dim_x:, dim_x:])


def prony_splitting(coeffs, dim_x):
    """
    Compute the Kernel under prony series form
    """
    Ahv = coeffs["A"][dim_x:, :dim_x]
    Avh = coeffs["A"][:dim_x, dim_x:]
    eigs, right_vect = np.linalg.eig(coeffs["A"][dim_x:, dim_x:])
    right_coeffs = np.linalg.inv(right_vect) @ Ahv
    left_coeffs = Avh @ right_vect
    a_est = -1 * left_coeffs[0, :] * right_coeffs[:, 0]
    return np.abs(a_est), -1 * eigs, np.angle(a_est)


def prony(t, F, m):
    """Input  : real arrays t, F of the same size (ti, Fi): integer m - the number of modes in the exponential fit
    Output : arrays a and b such that F(t) ~ sum ai exp(bi*t)
    """

    # Solve LLS problem in step 1
    # Amat is (N-m)*m and bmat is N-m*1
    N = len(t)
    Amat = np.zeros((N - m, m), dtype=np.complex)
    bmat = F[m:N]

    for jcol in range(m):
        Amat[:, jcol] = F[m - jcol - 1 : N - 1 - jcol]

    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    d = sol[0]

    # Solve the roots of the polynomial in step 2
    # first, form the polynomial coefficients
    c = np.zeros(m + 1, dtype=np.complex)
    c[m] = 1.0
    for i in range(1, m + 1):
        c[m - i] = -d[i - 1]

    u = poly.polyroots(c)
    b_est = np.log(u) / (t[1] - t[0])

    # Set up LLS problem to find the "a"s in step 3
    Amat = np.zeros((N, m), dtype=np.complex)
    bmat = F

    for irow in range(N):
        Amat[irow, :] = u ** irow

    sol = np.linalg.lstsq(Amat, bmat, rcond=None)
    a_est = sol[0]

    return np.abs(a_est), b_est, np.angle(a_est)


def prony_eval(t, a, b, c):
    """
    Evaluate a prony series for each point in t
    """
    series = np.zeros_like(t, dtype=np.complex)
    for i, a in enumerate(a):
        series += a[i] * np.exp(b[i] * t + 1j * c[i])
    return np.real(series)


def correlation(a, b=None, subtract_mean=False):
    """
    Correlation between a and b
    """
    meana = int(subtract_mean) * np.mean(a)
    a2 = np.append(a - meana, np.zeros(2 ** int(np.ceil((np.log(len(a)) / np.log(2)))) - len(a)))
    data_a = np.append(a2, np.zeros(len(a2)))
    fra = np.fft.fft(data_a)
    if b is None:
        sf = np.conj(fra) * fra
    else:
        meanb = int(subtract_mean) * np.mean(b)
        b2 = np.append(b - meanb, np.zeros(2 ** int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)))
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft(data_b)
        sf = np.conj(fra) * frb
    res = np.fft.ifft(sf)
    cor = np.real(res[: len(a)]) / np.array(range(len(a), 0, -1))
    return cor[: len(cor) // 2]


def forcefield(x_lims, basis, force_coeffs):
    """Compute and save the force field that have been fitted

    Parameters
    ----------

    x_lims: array, shape (dim_x,3)
        Bounds of the plot

    basis: GLE_BasisTransform instance
        The instance of the basis projection

    force_coeffs: array-like
        Value of the force coefficients in the basis

    """
    x_lims = np.asarray(x_lims)
    if x_lims.ndim == 1:  # In case of flat array
        x_lims.reshape(1, -1)
    if x_lims.shape[0] == 1:  # 1D:
        X = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2])).reshape(-1, 1)
    elif x_lims.shape[0] == 2:  # 2D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        x, y = np.meshgrid(x_coords, y_coords)
        X = np.vstack((x.flatten(), y.flatten())).T
    elif x_lims.shape[0] == 3:  # 3D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], int(x_lims[2][2]))
        x, y, z = np.meshgrid(x_coords, y_coords, z_coords)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    elif x_lims.shape[0] == 4:  # 4D:
        x_coords = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
        y_coords = np.linspace(x_lims[1][0], x_lims[1][1], int(x_lims[1][2]))
        z_coords = np.linspace(x_lims[2][0], x_lims[2][1], int(x_lims[2][2]))
        c_coords = np.linspace(x_lims[3][0], x_lims[3][1], int(x_lims[3][2]))
        x, y, z, c = np.meshgrid(x_coords, y_coords, z_coords, c_coords)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten(), c.flatten())).T
    elif x_lims.shape[0] > 4:
        raise NotImplementedError("Dimension higher than 3 are not implemented")
    force_field = np.matmul(force_coeffs, basis.transform(X).T).T
    return np.hstack((X, force_field))


def forcefield_plot2D(x_lims, basis, force_coeffs):
    """Compute and save the force field that have been fitted

    Parameters
    ----------

    x_lims: array, shape (dim_x,3)
        Bounds of the plot

    basis: GLE_BasisTransform instance
        The instance of the basis projection

    force_coeffs: array-like
        Value of the force coefficients in the basis

    """
    x_coords = np.linspace(x_lims[0][0], x_lims[0][1], x_lims[0][2])
    y_coords = np.linspace(x_lims[1][0], x_lims[1][1], x_lims[1][2])
    x, y = np.meshgrid(x_coords, y_coords)
    X = np.vstack((x.flatten(), y.flatten())).T
    force_field = np.matmul(force_coeffs, basis.transform(X).T).T
    f = force_field.reshape(x_lims[0][2], x_lims[1][2], 2)
    return x, y, f[:, :, 0], f[:, :, 1]


def potential_reconstruction_1D(x_lims, basis, force_coeffs):
    """From the force field reconstruct the potential"""
    x = np.linspace(x_lims[0][0], x_lims[0][1], int(x_lims[0][2]))
    sol = scipy.integrate.odeint(lambda y, t: -np.matmul(force_coeffs, basis.transform(np.array([t]).reshape(-1, 1)).T)[0, 0], [0.0], x)
    return x, sol[:, 0]
