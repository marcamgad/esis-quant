from itertools import product

import numpy as np

from models.hmm import GaussianHMM, _log_gaussian_pdf, _logsumexp


def _make_fixed_hmm(pi, A, mu, sigma, n_states):
    """Build a GaussianHMM with known parameters injected directly
    (bypassing fit/EM), so forward/backward/Viterbi can be validated
    against brute force independent of the EM training loop.
    """
    hmm = GaussianHMM(n_states=n_states)
    hmm.pi_ = np.asarray(pi)
    hmm.A_ = np.asarray(A)
    hmm.mu_ = np.asarray(mu)
    hmm.sigma_ = np.asarray(sigma)
    return hmm


def _gaussian_density_1d(x, mu, var):
    return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu) ** 2 / var)


def _brute_force_total_prob(X, pi, A, mu, sigma, n_states):
    T = X.shape[0]
    total = 0.0
    for path in product(range(n_states), repeat=T):
        p = pi[path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]]
        for t in range(T):
            p *= _gaussian_density_1d(X[t, 0], mu[path[t], 0], sigma[path[t], 0, 0])
        total += p
    return total


def _brute_force_best_path(X, pi, A, mu, sigma, n_states):
    T = X.shape[0]
    best_p, best_path = -1.0, None
    for path in product(range(n_states), repeat=T):
        p = pi[path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]]
        for t in range(T):
            p *= _gaussian_density_1d(X[t, 0], mu[path[t], 0], sigma[path[t], 0, 0])
        if p > best_p:
            best_p, best_path = p, path
    return best_path, best_p


def test_logsumexp_matches_naive_for_moderate_values():
    a = np.array([-10.0, -10.5, -9.0])
    naive = np.log(np.sum(np.exp(a)))  # doesn't underflow at this magnitude, safe as a reference
    assert np.isclose(_logsumexp(a), naive, atol=1e-10)


def test_logsumexp_handles_values_that_would_underflow_naively():
    a = np.array([-1000.0, -1000.5, -999.0])
    with np.errstate(divide="ignore"):
        naive = np.log(np.sum(np.exp(a)))  # exp(-1000) underflows to 0.0 -> log(0) = -inf
    assert naive == -np.inf  # confirms this case really does break the naive approach

    # Manually shift by the max (the trick itself) as an independent check,
    # rather than asserting _logsumexp against its own implementation.
    shift = -999.0
    expected = shift + np.log(np.exp(-1000.0 - shift) + np.exp(-1000.5 - shift) + np.exp(-999.0 - shift))
    assert np.isfinite(expected)
    assert np.isclose(_logsumexp(a), expected, atol=1e-10)


def test_logsumexp_all_neg_inf_returns_neg_inf_not_nan():
    a = np.array([-np.inf, -np.inf, -np.inf])
    assert _logsumexp(a) == -np.inf


def test_forward_score_matches_brute_force_enumeration():
    rng = np.random.default_rng(42)
    K = 2
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    mu = np.array([[0.0], [3.0]])
    sigma = np.array([[[1.0]], [[1.0]]])
    T = 6
    X = rng.normal(size=(T, 1))

    hmm = _make_fixed_hmm(pi, A, mu, sigma, K)
    loglik_forward = hmm.score(X)

    brute_total = _brute_force_total_prob(X, pi, A, mu, sigma, K)
    assert np.isclose(loglik_forward, np.log(brute_total), atol=1e-8)


def test_viterbi_matches_brute_force_best_path():
    rng = np.random.default_rng(7)
    K = 3
    pi = np.array([0.5, 0.3, 0.2])
    A = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    mu = np.array([[0.0], [2.0], [-2.0]])
    sigma = np.array([[[1.0]], [[1.5]], [[0.8]]])
    T = 5
    X = rng.normal(size=(T, 1))

    hmm = _make_fixed_hmm(pi, A, mu, sigma, K)
    path, loglik = hmm.decode(X)

    brute_path, brute_p = _brute_force_best_path(X, pi, A, mu, sigma, K)
    assert tuple(path) == brute_path
    assert np.isclose(loglik, np.log(brute_p), atol=1e-8)


def _sample_from_hmm(pi, A, mu, sigma, T, rng):
    K = len(pi)
    states = np.zeros(T, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for t in range(1, T):
        states[t] = rng.choice(K, p=A[states[t - 1]])
    d = mu.shape[1]
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = rng.multivariate_normal(mu[states[t]], sigma[states[t]])
    return X, states


def test_em_loglikelihood_is_monotonically_nondecreasing():
    rng = np.random.default_rng(3)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[0.0, 0.0], [4.0, 4.0]])
    sigma = np.array([np.eye(2), np.eye(2)])
    X, _ = _sample_from_hmm(pi, A, mu, sigma, T=300, rng=rng)

    hmm = GaussianHMM(n_states=2, n_init=1, random_state=0, tol=1e-8, n_iter=100)
    hmm.fit(X)

    diffs = np.diff(hmm.monitor_)
    assert (diffs > -1e-6).all(), f"log-likelihood decreased during EM: {hmm.monitor_}"


def test_em_recovers_known_two_state_regimes():
    rng = np.random.default_rng(11)
    true_pi = np.array([0.5, 0.5])
    true_A = np.array([[0.95, 0.05], [0.05, 0.95]])
    true_mu = np.array([[0.0, 0.0], [5.0, 5.0]])
    true_sigma = np.array([np.eye(2), np.eye(2)])
    X, true_states = _sample_from_hmm(true_pi, true_A, true_mu, true_sigma, T=500, rng=rng)

    hmm = GaussianHMM(n_states=2, n_init=5, random_state=1, tol=1e-6, n_iter=200)
    hmm.fit(X)
    decoded, _ = hmm.decode(X)

    # Resolve label-permutation ambiguity: match fitted states to true
    # states by nearest fitted mean, then remap before scoring accuracy.
    dist_to_true0 = np.linalg.norm(hmm.mu_ - true_mu[0], axis=1)
    dist_to_true1 = np.linalg.norm(hmm.mu_ - true_mu[1], axis=1)
    fitted_state_for_true0 = int(np.argmin(dist_to_true0))
    fitted_state_for_true1 = int(np.argmin(dist_to_true1))
    assert fitted_state_for_true0 != fitted_state_for_true1  # states must have actually differentiated

    remap = {fitted_state_for_true0: 0, fitted_state_for_true1: 1}
    remapped_decoded = np.array([remap[s] for s in decoded])

    accuracy = (remapped_decoded == true_states).mean()
    assert accuracy > 0.85, f"decoding accuracy too low: {accuracy}"


def test_covariance_regularization_prevents_singular_matrix_error():
    # Degenerate case: a state where all assigned observations are
    # identical would make the raw sample covariance singular; cov_reg
    # must keep _log_gaussian_pdf from raising.
    X = np.zeros((5, 2))
    mu = np.zeros(2)
    singular_sigma = np.zeros((2, 2))
    result = _log_gaussian_pdf(X, mu, singular_sigma, reg=1e-6)
    assert np.all(np.isfinite(result))


def test_filtered_proba_is_causal_and_normalized():
    # Filtered P(q_t | X_0..t) must be a proper distribution AND invariant to
    # future data: appending observations after t cannot change the filtered
    # probability at t (the defining property that makes it point-in-time safe).
    rng = np.random.default_rng(7)
    pi = np.array([0.6, 0.4])
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu = np.array([[0.0, 0.0], [3.0, 3.0]])
    sigma = np.stack([np.eye(2), np.eye(2)])
    hmm = _make_fixed_hmm(pi, A, mu, sigma, n_states=2)
    X, _ = _sample_from_hmm(pi, A, mu, sigma, T=200, rng=rng)

    full = hmm.filtered_proba(X)
    assert full.shape == (200, 2)
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-9)  # rows are distributions

    # causality: filtered prob at t computed on X[:t+1] equals the one from full X
    for t in [10, 50, 137, 199]:
        prefix = hmm.filtered_proba(X[: t + 1])
        assert np.allclose(prefix[t], full[t], atol=1e-9)


def test_filtered_decode_recovers_states_on_separated_means():
    rng = np.random.default_rng(8)
    pi = np.array([0.5, 0.5])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[-4.0, -4.0], [4.0, 4.0]])  # well-separated -> filtering is confident
    sigma = np.stack([np.eye(2), np.eye(2)])
    hmm = _make_fixed_hmm(pi, A, mu, sigma, n_states=2)
    X, true = _sample_from_hmm(pi, A, mu, sigma, T=400, rng=rng)
    dec = hmm.filtered_decode(X)
    acc = max((dec == true).mean(), (dec == 1 - true).mean())  # label-permutation invariant
    assert acc > 0.9
