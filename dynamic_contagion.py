from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from numba import njit
from numba.typed import List
from numba import types

from scipy.integrate import cumulative_trapezoid


@njit()
def dcp_intensity(t, lambda_0, a, delta, T1, X, T2, Y):
    """
    Calculate the intensity of the Dynamic Contagion Process (DCP) at a specific time.

    Args:
        t: Time point at which to calculate the intensity.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The intensity of the DCP at time t.
    """
    background = a + (lambda_0 - a) * np.exp(-delta * t)
    cox = 0.0
    for Xi, T1i in zip(X, T1):
        if T1i <= t:
            cox += Xi * np.exp(-delta * (t - T1i))
    hawkes = 0.0
    for Yj, T2j in zip(Y, T2):
        if T2j <= t:
            hawkes += Yj * np.exp(-delta * (t - T2j))
    return background + cox + hawkes


def dcp_intensities(time_grid, lambda_0, a, delta, T1, X, T2, Y):
    """
    Calculate the intensity of the Dynamic Contagion Process (DCP) over a time grid.

    Args:
        time_grid: Time points at which to calculate the intensity.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The intensity of the DCP at each time point in time_grid.
    """
    if len(T1) == 0:
        T1 = List.empty_list(types.float64)
    if len(X) == 0:
        X = List.empty_list(types.float64)
    if len(T2) == 0:
        T2 = List.empty_list(types.float64)
    if len(Y) == 0:
        Y = List.empty_list(types.float64)

    intensities = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        intensities[i] = dcp_intensity(t, lambda_0, a, delta, T1, X, T2, Y)

    return intensities


@njit()
def dcp_integrated_intensity(t, lambda_0, a, delta, T1, X, T2, Y):
    """
    Calculate the integrated intensity of the Dynamic Contagion Process (DCP) over [0, t].

    Args:
        t: Time point up to which to integrate the intensity.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The integrated intensity of the DCP over [0, t].
    """
    # Integrated background component
    background = (a / delta) * (1 - np.exp(-delta * t)) + (lambda_0 - a) * (
        1 - np.exp(-delta * t)
    ) / delta

    # Integrated Cox component
    cox = 0.0
    for Xi, T1i in zip(X, T1):
        if T1i < t:
            cox += Xi * (1 - np.exp(-delta * (t - T1i))) / delta

    # Integrated Hawkes component
    hawkes = 0.0
    for Yj, T2j in zip(Y, T2):
        if T2j < t:
            hawkes += Yj * (1 - np.exp(-delta * (t - T2j))) / delta

    return background + cox + hawkes


def dcp_integrated_intensities(time_grid, lambda_0, a, delta, T1, X, T2, Y):
    """
    Calculate the integrated intensity of the Dynamic Contagion Process (DCP) over a time grid.

    Args:
        time_grid: Time points at which to calculate the intensity.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The intensity of the DCP at each time point in time_grid.
    """
    if len(T1) == 0:
        T1 = List.empty_list(types.float64)
    if len(X) == 0:
        X = List.empty_list(types.float64)
    if len(T2) == 0:
        T2 = List.empty_list(types.float64)
    if len(Y) == 0:
        Y = List.empty_list(types.float64)

    intensities = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        intensities[i] = dcp_integrated_intensity(t, lambda_0, a, delta, T1, X, T2, Y)

    return intensities


@njit()
def dcp_max_intensity(t, max_time, lambda_0, a, delta, T1, X, T2, Y):
    """
    Calculate the maximum intensity of the Dynamic Contagion Process (DCP) from a specific time up to a maximum time.

    Args:
        t: Time point at which to calculate the maximum intensity.
        max_time: Maximum time to consider.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The maximum intensity of the DCP up to time t.
    """
    # Note, the maximum intensity could only be the current intensity (time t) or one of the subsequent jump times
    max_intensity = dcp_intensity(t, lambda_0, a, delta, T1, X, T2, Y)
    for T1i in T1:
        if t < T1i <= max_time:
            max_intensity = max(
                max_intensity, dcp_intensity(T1i, lambda_0, a, delta, T1, X, T2, Y)
            )

    return max_intensity


def simulate_dynamic_contagion(
    rg: np.random.Generator,
    max_time: float,
    lambda0: float,
    a: float,
    rho: float,
    delta: float,
    self_jump_size_dist: Callable[[np.random.Generator], float],
    ext_jump_size_dist: Callable[[np.random.Generator], float],
) -> tuple[int, list[float], list[float], list[float], list[float]]:
    """Simulate a dynamic contagion process and return the number of arrivals plus some internals.

    Args:
        rg: A random number generator.
        max_time: When to stop simulating.
        lambda0: The initial intensity at time t = 0.
        a: The constant mean-reverting level.
        rho: The rate of arrivals for the Poisson external jumps.
        delta: The rate of exponential decay in intensity.
        self_jump_size_dist: A function which samples intensity jump sizes for self-arrivals.
        ext_jump_size_dist: A function which samples intensity jump sizes for external-arrivals.

    Returns:
        count: The number of arrivals at time max_time.
        T1: The times of the external Poisson point process arrivals.
        X: The jump sizes for the external Poisson point process.
        T2: The times of the self-excited point process arrivals.
        Y: The jump sizes for the self-excited point process.
    """

    # Step 1: Set initial conditions
    prev_time = 0.0
    intensity = lambda0

    count = 0

    T1, X, T2, Y = [], [], [], []

    while True:
        # Step 2: Simulate the next externally excited jump waiting time
        E: float = (1 / rho) * rg.exponential()

        # Step 3: Simulate the next self-excited jump waiting time
        d: float = 1 - (delta * rg.exponential()) / (intensity - a)

        S1: float = -(1 / delta) * np.log(d) if d > 0 else float("inf")
        S2: float = (1 / a) * rg.exponential()

        S = min(S1, S2)

        # Step 4: Simulate the next jump time
        waiting_time = min(S, E)
        assert waiting_time > 0

        time = prev_time + waiting_time

        if time > max_time:
            break

        if S < E:
            # Self-excited jump
            count += 1
            jump_size = self_jump_size_dist(rg)
            T2.append(time)
            Y.append(jump_size)
        else:
            # External jump
            jump_size = ext_jump_size_dist(rg)
            T1.append(time)
            X.append(jump_size)

        # Step 5: Update the intensity process
        intensity_pre_jump: float = (intensity - a) * np.exp(-delta * waiting_time) + a
        intensity = intensity_pre_jump + jump_size

        prev_time = time

    return count, T1, X, T2, Y


def simulate_dynamic_contagion_thinning(
    rg: np.random.Generator,
    max_time: float,
    lambda_0: float,
    a: float,
    rho: float,
    delta: float,
    self_jump_size_dist: Callable[[np.random.Generator], float],
    ext_jump_size_dist: Callable[[np.random.Generator], float],
    step=10.0,
) -> tuple[int, list[float], list[float], list[float], list[float]]:
    """Simulate a dynamic contagion process and return the number of arrivals plus some internals.

    Args:
        rg: A random number generator.
        max_time: When to stop simulating.
        lambda_0: The initial intensity at time t = 0.
        a: The constant mean-reverting level.
        rho: The rate of arrivals for the Poisson external jumps.
        delta: The rate of exponential decay in intensity.
        self_jump_size_dist: A function which samples intensity jump sizes for self-arrivals.
        ext_jump_size_dist: A function which samples intensity jump sizes for external-arrivals.

    Returns:
        count: The number of arrivals at time max_time.
        T1: The times of the external Poisson point process arrivals.
        X: The jump sizes for the external Poisson point process.
        T2: The times of the self-excited point process arrivals.
        Y: The jump sizes for the self-excited point process.
    """

    # Step 1: Simulate external jump process (Cox process part)
    num_shots = rg.poisson(rho * max_time)
    T1 = rg.uniform(0, max_time, num_shots)
    T1.sort()
    X = [ext_jump_size_dist(rg) for _ in range(num_shots)]

    if len(T1) == 0:
        T1 = List.empty_list(types.float64)
    if len(X) == 0:
        X = List.empty_list(types.float64)
    T2 = List.empty_list(types.float64)
    Y = List.empty_list(types.float64)

    # Step 2: Simulate self-excited arrivals (Hawkes process part)
    t = 0
    while True:
        lambda_max = dcp_max_intensity(
            t, min(t + step, max_time), lambda_0, a, delta, T1, X, T2, Y
        )

        delta_t = rg.exponential(1 / lambda_max)

        t += min(delta_t, step)
        if t > max_time:
            break

        if delta_t > step:
            continue

        lambda_t_current = dcp_intensity(t, lambda_0, a, delta, T1, X, T2, Y)

        assert lambda_t_current <= lambda_max
        if rg.random() < lambda_t_current / lambda_max:
            T2.append(t)
            Y.append(self_jump_size_dist(rg))

    return len(T2), T1, X, T2, Y


def lambda_t_expectation(t, lambda_0, mu_H, rho, a, delta, mu_G):
    """
    Compute the conditional expectation of lambda_t given lambda_0.

    Args:
        t: Time at which to evaluate the expectation.
        lambda_0: Initial value of lambda at t=0.
        mu_H: Parameter mu_H.
        rho: Parameter rho.
        a: Parameter a.
        delta: Parameter delta.
        mu_G: Parameter mu_G.

    Returns:
        Conditional expectation of lambda_t.
    """
    kappa = delta - mu_G
    if kappa != 0:
        steady_state_mean = (mu_H * rho + a * delta) / kappa
        return steady_state_mean + (lambda_0 - steady_state_mean) * np.exp(-kappa * t)
    else:
        return lambda_0 + (mu_H * rho + a * delta) * t


def lambda_t_asymptotic_mean(mu_H, rho, a, delta, mu_G):
    """
    Compute the asymptotic first moment of lambda_t.

    Args:
        mu_H: Parameter mu_H.
        rho: Parameter rho.
        a: Parameter a.
        delta: Parameter delta.
        mu_G: Parameter mu_G.

    Returns:
        Asymptotic first moment of lambda_t.
    """
    kappa = delta - mu_G
    if kappa > 0:
        return (mu_H * rho + a * delta) / kappa
    else:
        raise ValueError("Stationary condition (kappa > 0) is not satisfied.")


def Nt_expectation(t, lambda_0, mu_H, rho, a, delta, mu_G):
    """
    Compute the conditional expectation of N_t given lambda_0.

    Args:
        t: Time at which to evaluate the expectation.
        lambda_0: Initial value of lambda at t=0.
        mu_H: Parameter mu_H.
        rho: Parameter rho.
        a: Parameter a.
        delta: Parameter delta.
        mu_G: Parameter mu_G.

    Returns:
        Conditional expectation of N_t.
    """
    kappa = delta - mu_G
    mu_1 = (mu_H * rho + a * delta) / kappa if kappa > 0 else None

    if kappa != 0:
        steady_state_mean = (mu_H * rho + a * delta) / kappa
        return (
            mu_1 * t + (lambda_0 - steady_state_mean) * (1 - np.exp(-kappa * t)) / kappa
        )
    else:
        return lambda_0 * t + 0.5 * (mu_H * rho + a * delta) * t**2


def tilted_dcp_intensities(t_start, time_grid, lambda_0, new_as, delta, T1, X, T2, Y):
    """
    Calculate the intensity of the Dynamic Contagion Process (DCP) over a time grid.

    Args:
        time_grid: Time points at which to calculate the intensity.
        lambda_0: Initial intensity at t=0.
        a: Constant mean-reverting level.
        delta: Constant rate of exponential decay.
        T1: Jump times for the Poisson point process.
        X: Jump sizes for the Poisson point process.
        T2: Jump times for the other point process.
        Y: Jump sizes for the other point process.

    Returns:
        The intensity of the DCP at each time point in time_grid.
    """

    if len(T1) == 0:
        T1 = List.empty_list(types.float64)
    if len(X) == 0:
        X = List.empty_list(types.float64)
    if len(T2) == 0:
        T2 = List.empty_list(types.float64)
    if len(Y) == 0:
        Y = List.empty_list(types.float64)

    intensities = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        if t < t_start:
            continue
        intensities[i] = dcp_intensity(t, lambda_0, new_as[i], delta, T1, X, T2, Y)

    return intensities


def simulate_tilted_dynamic_contagion_thinning(
    rg: np.random.Generator,
    max_time: float,
    lambda_0: float,
    new_as: list[float],
    new_rhos: list[float],
    new_delta: float,
    new_self_jump_size_dist: Callable[[np.random.Generator, int], float],
    new_ext_jump_size_dist: Callable[[np.random.Generator, int], float],
    time_grid: np.ndarray,
    step=10.0,
    plot=False,
) -> tuple[int, list[float], list[float], list[float], list[float]]:
    """Simulate a dynamic contagion process and return the number of arrivals plus some internals.

    Args:
        rg: A random number generator.
        max_time: When to stop simulating.
        lambda_0: The initial intensity at time t = 0.
        a: The constant mean-reverting level.
        rho: The rate of arrivals for the Poisson external jumps.
        delta: The rate of exponential decay in intensity.
        self_jump_size_dist: A function which samples intensity jump sizes for self-arrivals.
        ext_jump_size_dist: A function which samples intensity jump sizes for external-arrivals.

    Returns:
        count: The number of arrivals at time max_time.
        T1: The times of the external Poisson point process arrivals.
        X: The jump sizes for the external Poisson point process.
        T2: The times of the self-excited point process arrivals.
        Y: The jump sizes for the self-excited point process.
    """

    # Step 1: Simulate external jump process (Cox process part)
    max_rho = max(new_rhos)
    num_shots = rg.poisson(max_rho * max_time)
    t1_options = sorted(list(rg.uniform(0, max_time, num_shots)))
    X = []
    T1 = []

    for t1_i in t1_options:
        t_index = np.searchsorted(time_grid, t1_i)
        if rg.random() < new_rhos[t_index] / max_rho:
            T1.append(t1_i)
            X.append(new_ext_jump_size_dist(rg, t_index))

    if len(T1) == 0:
        T1 = List.empty_list(types.float64)
    if len(X) == 0:
        X = List.empty_list(types.float64)

    T2 = List.empty_list(types.float64)
    Y = List.empty_list(types.float64)

    cox_intensities = tilted_dcp_intensities(
        0, time_grid, lambda_0, new_as, new_delta, T1, X, T2, Y
    )
    if plot:
        plt.plot(time_grid, cox_intensities)
        plt.title("Cox part of intensity")
        plt.show()

    hawkes_intensities = np.zeros_like(time_grid)

    # Step 2: Simulate self-excited arrivals (Hawkes process part)
    t = 0
    while True:
        t_index = np.searchsorted(time_grid, t)
        lambda_max = (
            np.max(cox_intensities[t_index:] + hawkes_intensities[t_index:]) * 1.05
        )

        delta_t = rg.exponential(1 / lambda_max)

        t += min(delta_t, step)
        t_index = np.searchsorted(time_grid, t)
        if t > max_time:
            break

        if delta_t > step:
            continue

        lambda_t_current = dcp_intensity(
            t, lambda_0, new_as[t_index], new_delta, T1, X, T2, Y
        )

        assert lambda_t_current <= lambda_max
        if rg.random() < lambda_t_current / lambda_max:
            T2.append(t)
            Y.append(new_self_jump_size_dist(rg, t_index))
            hawkes_intensities[t_index:] += Y[-1] * np.exp(
                -new_delta * (time_grid[t_index:] - t)
            )

    if plot:
        plt.plot(time_grid, hawkes_intensities)
        plt.title("Hawkes part of intensity")
        plt.show()

        plt.plot(time_grid, cox_intensities + hawkes_intensities)
        plt.title("Total intensity")
        plt.show()

    return len(T2), list(T1), list(X), list(T2), list(Y)


def tilted_lambda_t_expectation(
    t, lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
):
    """
    Compute the conditional expectation of lambda_t given lambda_0 using the trapezoidal rule.

    Args:
        t: Time at which to evaluate the expectation.
        lambda_0: Initial value of lambda at t=0.
        new_as: List of a(t) values.
        new_rhos: List of rho(t) values.
        delta: Parameter delta.
        new_mu_Gs: List of mu_G(t) values.
        new_mu_Hs: List of mu_H(t) values.
        time_grid: Time grid for numerical integration.

    Returns:
        Conditional expectation of lambda_t.
    """
    if t == 0:
        return lambda_0

    dt = time_grid[1] - time_grid[0]

    # Select relevant time points up to t
    valid_time_points = time_grid[time_grid <= t]
    indices = np.searchsorted(time_grid, valid_time_points)

    # Compute the first integral using the trapezoidal rule
    kappa_t = delta - new_mu_Gs[indices]
    exp_integral = np.exp(-np.trapezoid(kappa_t, valid_time_points))

    # Compute the second integral using the trapezoidal rule
    integrand = new_rhos[indices] * new_mu_Hs[indices] + new_as[indices] * delta
    last_integral = np.trapezoid(
        np.exp(np.cumsum(kappa_t) * dt) * integrand, valid_time_points
    )

    return lambda_0 * exp_integral + exp_integral * last_integral


def tilted_N_t_expectation(
    t, lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
):
    """
    Compute the conditional expectation of N_t given lambda_0.

    Args:
        t: Time at which to evaluate the expectation.
        lambda_0: Initial value of lambda at t=0.
        new_as: List of a(t) values.
        new_rhos: List of rho(t) values.
        delta: Parameter delta.
        new_mu_Gs: List of mu_G(t) values.
        new_mu_Hs: List of mu_H(t) values.
        time_grid: Time grid for numerical integration.

    Returns:
        Conditional expectation of N_t.
    """
    # Select the time points up to t
    valid_time_points = time_grid[time_grid <= t]

    # Compute lambda expectations at valid time points
    lambda_vals = np.array(
        [
            tilted_lambda_t_expectation(
                s, lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
            )
            for s in valid_time_points
        ]
    )

    # Use the trapezoidal rule for integration
    integral_N = np.trapz(lambda_vals, valid_time_points)

    return integral_N


def tilted_N_t_expectations(
    lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
):
    # Compute the tilted lambda_t expectation once for every time point.
    lambda_vals = np.array(
        [
            tilted_lambda_t_expectation(
                t, lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
            )
            for t in time_grid
        ]
    )

    # Compute the cumulative integral using the trapezoidal rule.
    # cumulative_trapezoid returns an array with one fewer element unless we set initial=0.
    N_t_means = cumulative_trapezoid(lambda_vals, time_grid, initial=0)

    return N_t_means


def tilted_C_t_expectation(
    t, lambda_0, new_mu_J, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
):
    """
    Compute the conditional expectation of C_t given lambda_0.

    Args:
        t: Time at which to evaluate the expectation.
        lambda_0: Initial value of lambda at t=0.
        new_mu_J: Mean jump size parameter.
        new_as: List of a(t) values.
        new_rhos: List of rho(t) values.
        delta: Parameter delta.
        new_mu_Gs: List of mu_G(t) values.
        new_mu_Hs: List of mu_H(t) values.
        time_grid: Time grid for numerical integration.

    Returns:
        Conditional expectation of C_t.
    """
    return new_mu_J * tilted_N_t_expectation(
        t, lambda_0, new_as, new_rhos, delta, new_mu_Gs, new_mu_Hs, time_grid
    )
