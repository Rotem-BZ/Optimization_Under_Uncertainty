import time

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from typing import Tuple, List
from functools import partial
from collections import defaultdict
import cvxpy as cp
from tqdm import tqdm


def loss_coefficients(rho, alpha):
    a1 = -1
    b1 = rho
    a2 = -1 - rho / alpha
    b2 = rho * (1 - 1 / alpha)
    return a1, a2, b1, b2


def loss(x: Tuple[np.ndarray, float], xi_mat: np.ndarray, rho: float, alpha: float, average: bool = False):
    """

    Parameters
    ----------
    x
        tuple with the portfolio being tested, shape (d), and the tau value.
    xi_mat
        (n,d) matrix, each row is a xi vector we calcualte the loss for.
    rho, alpha
        parameters of the loss.
    average
        if True, return the average over the sample. if False, return the entire vector.
    Returns
    -------
    The loss defined in section 4.1, for each vector (row) in xi.
    """
    portfolio, tau = x
    inner_products = xi_mat @ portfolio  # shape (n)
    vec1 = -inner_products + rho * tau
    vec2 = inner_products * (-1 - rho / alpha) + rho * tau * (1 - 1 / alpha)
    losses = np.maximum(vec1, vec2)
    return (losses.mean(), {}) if average else (losses, {'optimal_k': (vec2 > vec1).astype(int)})


def offline_SAA_solver_func(xi_mat: np.ndarray, rho: float, alpha: float):
    portfolio = cp.Variable(xi_mat.shape[1])
    tau = cp.Variable()
    inner_products = xi_mat @ portfolio  # shape (n)
    vec1 = -inner_products + rho * tau
    vec2 = inner_products * (-1 - rho / alpha) + rho * tau * (1 - 1 / alpha)
    losses = cp.maximum(vec1, vec2)
    objective = cp.Minimize(cp.sum(losses) / xi_mat.shape[0])
    constraints = [portfolio >= 0, cp.sum(portfolio) == 1, tau >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return portfolio.value, tau.value


def online_SAA_solver_func(xi_mat: np.ndarray, rho: float, alpha: float, mu: float, initial_model: tuple = None):
    N, x_dim = xi_mat.shape
    if initial_model is not None:
        portfolio, tau = initial_model
    else:
        portfolio = np.ones(x_dim) * 1 / x_dim
        tau = 0.1
    a1, a2, b1, b2 = loss_coefficients(rho, alpha)
    for i, xi_i in enumerate(xi_mat):
        _, result_dict = loss(x=(portfolio, tau), xi_mat=xi_i.reshape(1, -1), rho=rho, alpha=alpha, average=False)
        opt_k = result_dict['optimal_k'].item()
        a = [a1, a2][opt_k]
        b = [b1, b2][opt_k]

        # update portfolio - NEG
        exp_vec = np.exp(-mu * a)
        portfolio = portfolio * exp_vec
        portfolio = portfolio / np.sum(portfolio)

        # update tau - SGD
        step_size = 1 / (i + 3)
        tau = tau - step_size * b

    return portfolio, tau


def loss_hat_bounded_case(x: Tuple[np.ndarray, float], xi_mat: np.ndarray, lamb: float, epsilon: float, rho: float,
                          alpha: float,
                          lower_bound: np.ndarray, upper_bound: np.ndarray, average: bool = False):
    """

    Parameters
    ----------
    x
        tuple with the portfolio being tested, shape (d), and the tau value.
    xi_mat
        (n,d) matrix, each row is a xi vector we calcualte the loss for.
    lamb, epsilon, rho, alpha
        parameters of the loss.
    lower_bound, upper_bound
        lower bound <= xi <= upper bound
    average
        if True, return the average over the sample. if False, return the entire vector.
    Returns
    -------
    The loss defined in section 4.1, for each vector (row) in xi.
    """
    x: np.ndarray

    portfolio, tau = x
    N = xi_mat.shape[0]
    a1, a2, b1, b2 = loss_coefficients(rho, alpha)
    k_loss_values = []
    xi_opts = []
    for a, b in [(a1, b1), (a2, b2)]:
        xi_opt = xi_mat.copy()
        xi_opt[:, a * portfolio > lamb] = upper_bound[a * portfolio > lamb]
        xi_opt[:, a * portfolio < -lamb] = lower_bound[a * portfolio < -lamb]
        values = a * (xi_opt @ portfolio) + b * tau - lamb * np.linalg.norm(xi_opt - xi_mat, ord=1, axis=1)
        k_loss_values.append(values)
        xi_opts.append(xi_opt)
    values_mat = np.stack(k_loss_values)
    losses = np.max(values_mat, axis=0) + lamb * epsilon
    optimal_k = np.argmax(values_mat, axis=0)
    indices = optimal_k * N + np.arange(N)
    optimal_xis = np.vstack(xi_opts)[indices]
    # optimal_xis = np.where(np.repeat(optimal_k.reshape(-1, 1), xi_mat.shape[0], axis=1), *xi_opts)
    return losses.mean() if average else losses, {'optimal_k': optimal_k, 'optimal_xis': optimal_xis}


# def offline_WDRO_solver(xi_mat: np.ndarray, N_iterations: int, rho: float, alpha: float, lamb: float,
#                                      epsilon: float, lower_bound: np.ndarray, upper_bound: np.ndarray):
#     x_dim = xi_mat.shape[1]
#     x = (np.ones(x_dim) * 1 / x_dim, 0.1)
#     for iteration in range(N_iterations):
#         # stage 1: for fixed x,tau find argmax xi
#         _, optimal_dict = loss_hat_bounded_case(x=x, xi_mat=xi_mat, lamb=lamb, epsilon=epsilon, rho=rho, alpha=alpha,
#                                                 lower_bound=lower_bound, upper_bound=upper_bound)
#         # optimal_k = optimal_dict['optimal_k']
#         optimal_xis = optimal_dict['optimal_xis']
#
#         # stage 2: for fixed xi, find argmin x, tau
#         portfolio = cp.Variable(xi_mat.shape[1])
#         tau = cp.Variable()
#
#         a1, a2, b1, b2 = loss_coefficients(rho, alpha)
#         k_1_vec = optimal_xis @ portfolio * a1 + tau * b1
#         k_2_vec = optimal_xis @ portfolio * a2 + tau * b2
#         final_vec = cp.maximum(k_1_vec, k_2_vec)
#         objective = cp.Minimize(cp.sum(final_vec) / xi_mat.shape[0])
#         constraints = [portfolio >= 0, cp.sum(portfolio) == 1, tau >= 0]
#         prob = cp.Problem(objective, constraints)
#         prob.solve()
#         x = (portfolio.value, tau.value)
#     return x


def offline_WDRO_solver(xi_mat: np.ndarray, rho: float, alpha: float,
                        epsilon: float, lower_bound: np.ndarray, upper_bound: np.ndarray):
    N, p_dim = xi_mat.shape
    K = 2
    a1, a2, b1, b2 = loss_coefficients(rho, alpha)
    lamb = cp.Variable()
    portfolio = cp.Variable(p_dim)
    tau = cp.Variable()
    s = cp.Variable(N)
    z_variables = [[cp.Variable(p_dim) for _ in range(K)] for __ in range(N)]
    v_variables = [[cp.Variable(p_dim) for _ in range(K)] for __ in range(N)]

    obj = cp.Minimize(lamb * epsilon + cp.sum(s) / N)
    constraints = [portfolio >= 0, cp.sum(portfolio) == 1, tau >= 0]
    for i in range(N):
        xi_i = xi_mat[i]
        z_list = z_variables[i]
        v_list = v_variables[i]
        for k in range(K):
            z = z_list[k]
            v = v_list[k]
            a = [a1, a2][k]
            b = [b1, b2][k]
            # conjugate of linear function constrained to be zero
            constraints.append(z - v == -a * portfolio)

            # the rest of the first constraint
            conjugate_value = b * tau
            support_function = cp.minimum(v, 0) @ lower_bound + cp.maximum(v, 0) @ upper_bound
            constraints.append(conjugate_value + support_function - z @ xi_i <= s[i])

            # second constraint - inf-norm
            constraints.append(cp.norm(z, "inf") <= lamb)

    prob = cp.Problem(obj, constraints)
    result = prob.solve()
    return portfolio.value, tau.value


def OMD_WDRO_bounded_case(xi_mat: np.ndarray, rho: float, alpha: float, lamb: float,
                          epsilon: list or float, lower_bound: np.ndarray, upper_bound: np.ndarray, eta_tilde: float,
                          initial_model: tuple = None):
    assert eta_tilde > 0
    x_dim = xi_mat.shape[1]
    xi_bar = np.max(np.abs(np.concatenate([lower_bound, upper_bound]))).item()
    if isinstance(epsilon, float):
        epsilon = [epsilon] * xi_mat.shape[0]
    elif isinstance(epsilon, tuple):
        C, c = epsilon
        epsilon = c * np.power(np.arange(1, xi_mat.shape[0] + 1), -1 / C)
    L = lamb
    if initial_model is not None:
        portfolio, tau = initial_model
    else:
        portfolio = np.ones(x_dim) * 1 / x_dim
        tau = 0.1
    a1, a2, b1, b2 = loss_coefficients(rho, alpha)
    for i, xi_i in enumerate(xi_mat):
        epsilon_i = epsilon[i]
        _, optimal_dict = loss_hat_bounded_case(x=(portfolio, tau), xi_mat=xi_i.reshape((1, -1)), lamb=lamb,
                                                epsilon=epsilon_i, rho=rho, alpha=alpha,
                                                lower_bound=lower_bound, upper_bound=upper_bound)
        optimal_k = optimal_dict['optimal_k'].item()
        a = [a1, a2][optimal_k]
        b = [b1, b2][optimal_k]
        optimal_xi = optimal_dict['optimal_xis'].reshape(-1)
        eta = eta_tilde / math.sqrt(i + 1)
        portfolio = portfolio * np.exp(-optimal_xi * a * eta)
        portfolio = portfolio / portfolio.sum()
        tau = max(2 * xi_bar * (alpha + rho) / (rho * (alpha - 1)),
                  min(xi_bar * (2 * alpha + rho) / (alpha * rho), tau - eta * b))
        lamb = max(0.0, min(L, lamb - eta * (epsilon_i - np.linalg.norm(optimal_xi - xi_i))))
    return portfolio, tau


# def OGD_SAA_bounded_case(xi_mat: np.ndarray, rho: float, alpha: float,
#                          lower_bound: np.ndarray, upper_bound: np.ndarray, eta_tilde: float):
#     """ This is just SGD """
#     assert eta_tilde > 0
#     eta = eta_tilde
#     x_dim = xi_mat.shape[1]
#     theta_p = np.zeros(x_dim)
#     portfolio = None
#     theta_tau = 0
#     a1, a2, b1, b2 = loss_coefficients(rho, alpha)
#     for i, xi_i in enumerate(xi_mat):  # online optimization starts here
#         eta = eta_tilde / math.sqrt(i+1)
#         portfolio = np.clip(theta_p / eta, lower_bound, upper_bound)
#         _, optimal_dict = loss(x=(portfolio, theta_tau / eta), xi_mat=xi_i.reshape(1, -1),
#                                rho=rho, alpha=alpha, average=False)
#         optimal_k = optimal_dict['optimal_k'].item()
#         a = [a1, a2][optimal_k]
#         b = [b1, b2][optimal_k]
#
#         # updates
#         theta_p -= a
#         theta_tau -= b
#     return portfolio, theta_tau / eta


# def OMD_WDRO_unbounded_case(xi_mat: np.ndarray, rho: float, alpha: float, epsilon: list or float,
#                             lower_bound: np.ndarray, upper_bound: np.ndarray, eta_tilde: float):
#     assert eta_tilde > 0
#     x_dim = xi_mat.shape[1]
#     xi_bar = np.max(np.abs(np.concatenate([lower_bound, upper_bound]))).item()
#     if isinstance(epsilon, float):
#         epsilon = [epsilon] * xi_mat.shape[0]
#     portfolio = np.ones(x_dim) * 1 / x_dim
#     tau = 0.1
#     a1, a2, b1, b2 = loss_coefficients(rho, alpha)
#     for i, xi_i in enumerate(xi_mat):  # online optimization starts here
#         epsilon_i = epsilon[i]
#         _, optimal_dict = loss_hat_bounded_case(x=(portfolio, tau), xi_mat=xi_i.reshape((1, -1)), lamb=0,
#                                                 epsilon=epsilon_i, rho=rho, alpha=alpha,
#                                                 lower_bound=lower_bound, upper_bound=upper_bound)
#         optimal_k = optimal_dict['optimal_k'].item()
#         j_star = np.argmax(np.abs(portfolio))
#         eta = eta_tilde / math.sqrt(i + 1)
#         z = np.sign(portfolio)[j_star] * epsilon_i - xi_i
#
#         # updating y
#         portfolio = portfolio * np.exp(-z * eta)
#         portfolio = portfolio / portfolio.sum()
#
#         # updating tau
#         b = [b1, b2][optimal_k]
#         tau = max(xi_bar * abs(alpha - rho) / (rho * (alpha - 1)), min(xi_bar / rho, tau - eta * b))
#     return portfolio, tau


def get_model_funcs(rho, alpha, lamb, epsilon_float, epsilon_list, eta_tilde, lower, upper):
    return {'SAA': partial(offline_SAA_solver_func, rho=rho, alpha=alpha),
            'WDRO_bounded': partial(offline_WDRO_solver, rho=rho, alpha=alpha, epsilon=epsilon_float,
                                    lower_bound=lower, upper_bound=upper),
            'OSAA': partial(online_SAA_solver_func, rho=rho, alpha=alpha, mu=eta_tilde),
            'OWDRO_bounded': partial(OMD_WDRO_bounded_case, rho=rho, alpha=alpha, lamb=lamb,
                                     epsilon=epsilon_list, upper_bound=upper, lower_bound=lower,
                                     eta_tilde=eta_tilde),
            }


def plot_func(metrics: list, model: str, datas, rho=None, alpha=None, lamb=None, epsilon_list=None,
              epsilon_float=None, upper=None, lower=None, eta_tilde=None, mu=None, test_data=None,
              show_fig: bool = True, ax=None, dashed_line: bool = False, model_label: bool = False):
    model_funcs = get_model_funcs(rho, alpha, lamb, epsilon_float, epsilon_list, eta_tilde, lower, upper)
    metric_colors = {'in-sample': 'b', 'out-of-sample': 'r', 'online-out-of-sample': 'g'}
    model_colors = {'SAA': 'c', 'WDRO_bounded': 'r', 'OSAA': 'purple'}
    scores = defaultdict(list)
    for data in tqdm(datas):
        trained_model = model_funcs[model](xi_mat=data)
        for metric in metrics:
            # if model == 'OSAA' and len(data) < 200:
            #     scores[metric].append(0)
            #     continue

            # if metric == 'online-out-of-sample':
            #     continue
            # if len(data)<200:
            #     scores[metric].append(0)
            #     continue

            # if dashed_line and metric == 'online-out-of-sample':
            #     continue
            # if dashed_line and len(data) < 200:
            #     scores[metric].append(0)
            #     continue
            if metric == 'in-sample':
                if metric in ['SAA', 'OSAA']:
                    value = loss(x=trained_model, xi_mat=data, rho=rho, alpha=alpha, average=True)[0]
                else:  # assume bounded case
                    value = loss_hat_bounded_case(x=trained_model, xi_mat=data, lamb=lamb, epsilon=epsilon_float,
                                                  rho=rho, alpha=alpha, lower_bound=lower, upper_bound=upper,
                                                  average=True)[0]
            elif metric == 'out-of-sample':
                assert test_data is not None, "out-of-sample requires a test set"
                value = loss(x=trained_model, xi_mat=test_data, rho=rho, alpha=alpha, average=True)[0]
            elif metric == 'online-out-of-sample':
                assert model.startswith('O'), "cannot compute online-out-of-sample for offline models"
                assert test_data is not None, "online-out-of-sample requires a test set"
                value = 0
                for xi in test_data:
                    trained_model = model_funcs[model](xi_mat=xi.reshape((1, -1)), initial_model=trained_model)
                    value += loss(x=trained_model, xi_mat=xi.reshape(1, -1), rho=rho, alpha=alpha, average=True)[0]
                value = value / test_data.shape[0]
            else:
                raise ValueError(
                    f"illegal metric {metric}. options are in-sample, out-of-sample and online-out-of-sample")
            scores[metric].append(value)

    if ax is None:
        ax = plt.gca()
    for metric, values in scores.items():
        if dashed_line:
            ax.plot(list(map(len, datas)), values, '--',
                    label=model if model_label else metric,
                    c=model_colors[model] if model_label else metric_colors[metric])
        else:
            ax.plot(list(map(len, datas)), values,
                    label=model if model_label else metric,
                    c=model_colors[model] if model_label else metric_colors[metric])
    if show_fig:
        plt.title(f"metrics of {model}")
        plt.legend()
        plt.show()


def new_data_generation_method(n: int):
    M = np.random.normal(0, 0.5, size=(n,n))
    M = np.zeros((n,n))
    np.fill_diagonal(M, 0.02)
    cov = M @ M.T
    rng = np.random.default_rng()
    psi = rng.multivariate_normal([0]*n, cov)
    return psi.reshape(-1, 1)


def generate_datas(d: int, Ns: List[int] or Tuple[int, int], N_test: int = None, new_generation_method: bool = False):
    if isinstance(Ns, tuple):
        assert len(Ns) == 2
        Ns = [Ns[0]] * Ns[1]
    lower = np.array([0.03 * i - 3 * 0.025 * i for i in range(1, d + 1)])
    upper = np.array([0.03 * i + 3 * 0.025 * i for i in range(1, d + 1)])
    if new_generation_method:
        psi = new_data_generation_method
    else:
        psi = lambda n: np.random.normal(0, 0.02, size=n).reshape(-1, 1)
    zeta = lambda n: np.stack(
        [np.clip(np.array([np.random.normal(0.03 * i, 0.025 * i) for i in range(1, d + 1)]), lower, upper) for _ in
         range(n)])
    test_data = zeta(N_test) + psi(N_test) if N_test is not None else None
    return [zeta(N) + psi(N) for N in Ns], test_data, upper, lower


def figure41():
    # plt.style.use("seaborn")
    d = 10
    Ns = np.ceil(np.logspace(0, 3, 20)).astype(int).tolist()
    datas, test_data, upper, lower = generate_datas(d, Ns, N_test=10_000)
    lamb = 0.5  # ??
    eps_for_lhat = 0.1  # ??
    alpha = 0.2
    rho = 5

    # c_range = [0.01, 0.02, 0.05, 0.1]
    # C_range = [2, 5, 10]
    c_range = [0.01, 0.02]
    C_range = [2, 5, 10]
    fig, axs = plt.subplots(len(C_range), len(c_range))
    for i in range(axs.size):
        C_index, c_index = np.unravel_index(i, axs.shape)
        C = C_range[C_index]
        c = c_range[c_index]
        ax = axs[C_index, c_index]
        plot_func(['in-sample', 'out-of-sample', 'online-out-of-sample'], 'OWDRO_bounded', datas, test_data=test_data,
                  rho=rho, alpha=alpha, lamb=lamb, epsilon_list=(C, c), upper=upper, lower=lower,
                  epsilon_float=eps_for_lhat, show_fig=False, ax=ax,
                  eta_tilde=0.01, dashed_line=False
                  )
        plot_func(['in-sample', 'out-of-sample', 'online-out-of-sample'], 'OWDRO_bounded', datas, test_data=test_data,
                  rho=rho, alpha=alpha, lamb=lamb, epsilon_list=(C, c), upper=upper, lower=lower,
                  epsilon_float=eps_for_lhat, show_fig=False, ax=ax,
                  eta_tilde=0.1, dashed_line=True
                  )
        if C_index == 0:
            ax.set_title(f"{c=}")
        if c_index == 0:
            ax.set_ylabel(f"{C=}")

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=1),
                    Line2D([0], [0], linestyle='--', color='gray', lw=1)]

    plt.legend(custom_lines, ['eta=0.01', 'eta=0.1'])

    fig.suptitle("Comparison of C,c,eta values in oWDRO")
    fig.tight_layout()
    plt.show()


def figure42():
    d = 10
    Ns = np.ceil(np.logspace(0, 4, 20)).astype(int).tolist()
    datas, test_data, upper, lower = generate_datas(d, Ns, N_test=10_000)
    eps_for_lhat = 0.1  # ??
    lamb = 0.5  # ??
    alpha = 0.2
    rho = 5
    mu = 0.3  # ??

    plot_func(['in-sample', 'out-of-sample', 'online-out-of-sample'], 'OSAA', datas, test_data=test_data,
              rho=rho, alpha=alpha, upper=upper, lower=lower, mu=mu, lamb=lamb,
              epsilon_float=eps_for_lhat, show_fig=False,
              eta_tilde=0.1, dashed_line=True
              )
    datas, test_data, upper, lower = generate_datas(d, Ns, N_test=50)
    plot_func(['in-sample', 'out-of-sample', 'online-out-of-sample'], 'OSAA', datas, test_data=test_data,
              rho=rho, alpha=alpha, upper=upper, lower=lower, mu=mu, lamb=lamb,
              epsilon_float=eps_for_lhat, show_fig=False,
              eta_tilde=0.01, dashed_line=False
              )
    plt.show()


def figure43(new_genration: bool = False):
    d = 10
    Ns = np.ceil(np.logspace(0, 3, 20)).astype(int).tolist()
    t0 = time.perf_counter()
    datas, test_data, upper, lower = generate_datas(d, Ns, N_test=1_000, new_generation_method=new_genration)
    print(f"{len(datas)=}, {datas[0].shape=}")
    t1 = time.perf_counter()
    print(f"time to create datasets: {t1 - t0} sec")
    lamb = 0.5  # ??
    eps_for_lhat = 0.1  # ??
    alpha = 0.2
    rho = 5

    for model in ['SAA', 'WDRO_bounded']:
        t0 = time.perf_counter()
        plot_func(['out-of-sample'], model, datas, test_data=test_data,
                  rho=rho, alpha=alpha, upper=upper, lower=lower, lamb=lamb,
                  epsilon_float=eps_for_lhat, show_fig=False,
                  eta_tilde=0.01, dashed_line=False, model_label=True)
        t1 = time.perf_counter()
        print(f"time to calculate {model}: {t1 - t0} sec")
    # plt.title(f"{new_genration=}")
    plt.legend()
    plt.show()


def table41():
    d = 10
    Ns = [10, 100, 1_000, 10_000]
    datas, _, upper, lower = generate_datas(d, Ns)
    lamb = 0.5  # ??
    eps_for_lhat = 0.1  # ??
    alpha = 0.2
    rho = 5
    eta_tilde = 0.01

    model_funcs = get_model_funcs(rho, alpha, lamb, eps_for_lhat, eps_for_lhat, eta_tilde, lower, upper)
    models_list = ['SAA', 'WDRO_bounded', 'OSAA', 'OWDRO_bounded']
    results = defaultdict(list)     # keys: model names, values: list of times in the order of Ns
    for model in models_list:
        model_func = model_funcs[model]
        print(f"model: {model}")
        for N, data in zip(Ns, datas):
            if N == 10_000 and model == 'WDRO_bounded':
                results[model].append(500.0)
                print(f"\t {N=}: time>500 (sec)")
                continue
            t0 = time.perf_counter()
            _ = model_func(xi_mat=data)
            t1 = time.perf_counter()
            results[model].append(t1 - t0)
            print(f"\t {N=}: time={t1 - t0} (sec)")
    df = pd.DataFrame(results)
    df['N'] = Ns
    df.set_index('N', drop=True, inplace=True)
    print(df)


def experiment_try():
    d = 10
    rho = 5
    alpha = 0.2
    lower = np.array([0.03 * i - 3 * 0.025 * i for i in range(1, d + 1)])
    upper = np.array([0.03 * i + 3 * 0.025 * i for i in range(1, d + 1)])
    psi = lambda n: np.random.normal(0, 0.02, size=n).reshape(-1, 1)
    zeta = lambda n: np.stack(
        [np.clip(np.array([np.random.normal(0.03 * i, 0.025 * i) for i in range(1, d + 1)]), lower, upper) for _ in
         range(n)])
    # Ns = [50, 100, 150]
    Ns = np.arange(50, 5_000, 50)

    datas = [zeta(N) + psi(N) for N in Ns]
    metrics = ['in-sample', 'out-of-sample', 'online-out-of-sample'][:2]
    model = ['SAA', 'WDRO_bounded', 'OSAA', 'OWDRO_bounded'][0]
    lamb = 0.5
    c = 0.5
    C = 2
    epsilon_list = (C, c)
    epsilon_float = 0.3
    eta_tilde = 0.1
    mu = 0.5
    N_test = 150
    test_data = zeta(N_test) + psi(N_test)

    plot_func(metrics, model, datas, rho, alpha, lamb, epsilon_list, epsilon_float,
              upper, lower, eta_tilde, mu, test_data=test_data)

    # SAA_portfolio, SAA_tau = offline_SAA_solver_func(data, rho, alpha)
    # print(SAA_portfolio.shape, SAA_tau.shape)

    # N_iterations = 5
    # WDRO_portfolio, WDRO_tau = offline_WDRO_solver(data, N_iterations, rho, alpha, lamb=0.2, epsilon=1,
    #                                                             lower_bound=lower, upper_bound=upper)
    # print(WDRO_portfolio.shape, WDRO_tau.shape)

    # online_WDRO_portfolio, online_WDRO_tau = OMD_WDRO_bounded_case(xi_mat=data, rho=rho, alpha=alpha, lamb=0.2,
    #                                                                epsilon=1.0, lower_bound=lower,
    #                                                                upper_bound=upper, eta_tilde=0.5)
    # print(online_WDRO_portfolio.shape, online_WDRO_tau)

    # unbounded_online_WDRO_portfolio, unbounded_online_WDRO_tau = OMD_WDRO_unbounded_case(xi_mat=data,
    #                                                                                      rho=rho,
    #                                                                                      alpha=alpha,
    #                                                                                      epsilon=1.0,
    #                                                                                      lower_bound=lower,
    #                                                                                      upper_bound=upper,
    #                                                                                      eta_tilde=0.5)
    # print(unbounded_online_WDRO_portfolio.shape, unbounded_online_WDRO_tau)

    # OGD_SAA_bounded_portfolio, OGD_SAA_bound_tau = OGD_SAA_bounded_case(xi_mat=data, rho=rho, alpha=alpha,
    #                                                                     lower_bound=lower, upper_bound=upper,
    #                                                                     eta_tilde=0.4)
    # print(OGD_SAA_bounded_portfolio.shape, OGD_SAA_bound_tau)

    # offline_WDRO_p, offline_WDRO_tau = offline_WDRO_solver(xi_mat=data, rho=rho, alpha=alpha, epsilon=0.5,
    #                                                        lower_bound=lower, upper_bound=upper)


def main():
    # experiment_try()
    # figure41()
    # figure42()
    figure43(new_genration=True)
    # table41()


if __name__ == '__main__':
    main()
