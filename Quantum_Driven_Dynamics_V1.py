# Quantum_Driven_Dynamics



# Quantum_Driven_Dynamics_V1.py
#
# IDEJA:
# Ulazna kola pokrecu disipativnu kvantnu memoriju.
# Naizmenicno se primenjuju kolektivna unitarna operacija,
# fazni udar odredjen prethodno poznatim kolom i unos tog kola.
#
# Porede se:
# 1. memorija bez faznih udara;
# 2. memorija sa faznim udarima;
# 3. fazni udari modulirani logistickom dinamikom.
#
# IZLAZ:
# Sedam sest-qubitnih registara, pomocni memorijski qubit
# i uslovne operacije koje daju zajednicku amplitudu sedmorke.
#
# Za sortiranu sedmorku S=(i1,...,i7):
#   v(S) = M_i7 ... M_i1 b
#   P(S) = ||v(S)||^2 / Z
#
# Z je zbir preko svih sedmorki. Racuna se tacno dinamickim
# programiranjem pomocu prenosa matrica gustine.
#
# M_i = RY(alpha_i) diag(cos(beta0_i), cos(beta1_i)).
# To je blok kvantnog kola dobijen uslovljavanjem pomocnog
# qubita na |0>. Matrice M_i u opstem slucaju ne komutiraju.
#
# Nema Statevector -> 7x39 projekcije.
# Nema sedam nezavisnih regresiranih brojeva.
# Nema slucajnih kombinacija, klasifikatora ili PyMC zavisnosti.
#
# Kvantno: unitarna memorija, fazni udari, uslovne amplitude.
# Klasicno: optimizacija, normalizacija i pretraga maksimuma.
#
# Granica tumacenja:
# Ovo je prediktivna adaptacija pokretane kvantne dinamike.
# Ne tvrdi se da su haos ili kvantna prednost dokazani.
#
# Osnova ideja:
# Robert May: https://doi.org/10.1038/261459a0
# Fujii / Nakajima: https://arxiv.org/abs/1602.08159



from pathlib import Path
from itertools import combinations
from math import comb
from time import perf_counter
import random
import sys

import numpy as np

random.seed(39)
np.random.seed(39)

try:
    from scipy.optimize import minimize
    from numba import njit, prange, get_num_threads, set_num_threads
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, DiagonalGate
    from qiskit.quantum_info import Statevector, Operator
except ImportError as exc:
    raise SystemExit(
        "Nedostaje zavisnost: "
        + str(exc)
        + ". Potrebni su NumPy, SciPy, Numba i Qiskit."
    )

set_num_threads(min(4, get_num_threads()))



FILES = (
    Path("/data/loto7_4724_k93.csv"),
    Path("/data/loto7_4724_k93_loto_2984.csv"),
    Path("/data/loto7_4724_k93_loto_plus_1740.csv"),
)



N = 39
K = 7
MEMORY_INPUT = 0.20
COLLECTIVE_ANGLE = 0.90
LOGISTIC_R = 3.80
LOGISTIC_INPUT = 0.20

CONFIGURATIONS = (
    ("memorija", 0.0, False),
    ("fazni udari", 2.0, False),
    ("logisticka modulacija", 2.0, True),
)

REGULARIZATION = 0.001
GRADIENT_TOLERANCE = 2e-5
MAX_ITERATIONS = 500
SEARCH_TOLERANCE = 1e-10
DETAILS = False

INITIAL_BOND_ANGLE = 0.37


def load_csv(path):
    if not path.is_file():
        raise ValueError("CSV nije pronadjen: " + str(path))

    data = None
    for delimiter in (",", ";", None):
        try:
            candidate = np.loadtxt(
                path, delimiter=delimiter, dtype=np.float64, ndmin=2
            )
            if candidate.shape[1] == K:
                data = candidate
                break
        except ValueError:
            pass

    if data is None:
        raise ValueError("CSV mora imati sedam numerickih kolona bez hedera.")
    if len(data) < 120:
        raise ValueError("Potrebno je najmanje 120 kola.")
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV sadrzi neispravne vrednosti.")
    if not np.all(data == np.rint(data)):
        raise ValueError("CSV sadrzi brojeve koji nisu celi.")
    if np.any((data < 1) | (data > N)):
        raise ValueError("Brojevi moraju biti od 1 do 39.")

    data = np.sort(data.astype(np.int64), axis=1)
    if np.any(np.diff(data, axis=1) <= 0):
        raise ValueError("Kolo sadrzi duplikate.")

    # Hronoloski redosled redova ostaje nepromenjen.
    return data


def unitary_memory_step(rho, present, strength):
    """
    R = I + (exp(-i*p)-1)|u><u|
    U = D(present, strength) R

    Racunanje koristi rang-jedan oblik R, bez pune
    matricne multiplikacije 39x39 za svako kolo.
    """
    n = len(present)
    u = np.ones(n) / np.sqrt(n)
    coefficient = np.exp(-1j * COLLECTIVE_ANGLE) - 1.0

    rho_u = rho @ u
    overlap = float(np.real(np.vdot(u, rho_u)))

    evolved = (
        rho
        + coefficient * np.outer(u, rho_u.conj())
        + coefficient.conjugate() * np.outer(rho_u, u)
        + abs(coefficient) ** 2 * overlap * np.outer(u, u)
    )

    phase = np.exp(-1j * strength * (present - present.mean()))
    return evolved * phase[:, None] * phase.conj()[None, :]


def make_history(draws, kick_strength, logistic):
    total = len(draws)
    features = np.empty((total, N, 4))

    u = np.ones(N) / np.sqrt(N)
    rho = np.outer(u, u).astype(np.complex128)
    previous_diagonal = np.real(np.diag(rho)).copy()
    previous_present = np.zeros(N)
    logistic_state = 0.5

    for t, draw in enumerate(draws):
        present = np.zeros(N)
        present[draw - 1] = 1.0

        if logistic:
            drive = 1.0 - np.dot(present, previous_present) / K
            logistic_state = (
                (1.0 - LOGISTIC_INPUT)
                * LOGISTIC_R
                * logistic_state
                * (1.0 - logistic_state)
                + LOGISTIC_INPUT * drive
            )
            if not 0.0 <= logistic_state <= 1.0:
                raise RuntimeError("Logisticko stanje je izvan opsega.")
            strength = kick_strength * (0.75 + 0.5 * logistic_state)
        else:
            strength = kick_strength

        evolved = unitary_memory_step(rho, present, strength)
        input_state = present / np.sqrt(K)
        rho = (
            (1.0 - MEMORY_INPUT) * evolved
            + MEMORY_INPUT * np.outer(input_state, input_state)
        )

        diagonal = np.real(np.diag(rho))
        coherence = np.real(rho.sum(axis=1))
        coherence -= coherence.mean()

        features[t, :, 0] = 1.0
        features[t, :, 1] = np.tanh(N * diagonal - 1.0)
        features[t, :, 2] = np.tanh(coherence)
        features[t, :, 3] = np.tanh(
            N * (diagonal - previous_diagonal)
        )

        previous_diagonal = diagonal.copy()
        previous_present = present

    if not np.isclose(np.trace(rho), 1.0, atol=1e-9):
        raise RuntimeError("Trag kvantne memorije nije jedan.")
    if not np.allclose(rho, rho.conj().T, atol=1e-10):
        raise RuntimeError("Kvantna memorija nije hermitska.")
    if np.linalg.eigvalsh(rho).min() < -1e-9:
        raise RuntimeError("Kvantna memorija nije pozitivna.")

    return (
        np.ascontiguousarray(features[:-1]),
        np.ascontiguousarray(draws[1:] - 1),
        np.ascontiguousarray(features[-1]),
    )


@njit
def local_transfer(parameters, feature):
    alpha = (
        parameters[0]
        + parameters[1] * feature[1]
        + parameters[2] * feature[2]
    )
    field = (
        parameters[3]
        + parameters[4] * feature[1]
        + parameters[5] * feature[3]
    )
    h = np.tanh(field)
    beta0 = 0.7 + 0.4 * h
    beta1 = 0.7 - 0.4 * h

    ca = np.cos(alpha / 2.0)
    sa = np.sin(alpha / 2.0)
    d0 = np.cos(beta0)
    d1 = np.cos(beta1)

    a = ca * d0
    b = -sa * d1
    c = sa * d0
    d = ca * d1

    matrix = np.array([[a, b], [c, d]])

    # Vektor simetricne matrice gustine: [rho00, rho01, rho11].
    transfer = np.array([
        [a * a, 2.0 * a * b, b * b],
        [a * c, a * d + b * c, b * d],
        [c * c, 2.0 * c * d, d * d],
    ])

    matrix_derivatives = np.empty((6, 4))
    angle_features = np.array([1.0, feature[1], feature[2]])
    filter_features = np.array([1.0, feature[1], feature[3]])

    for p in range(3):
        factor = angle_features[p]
        matrix_derivatives[p, 0] = -0.5 * sa * d0 * factor
        matrix_derivatives[p, 1] = -0.5 * ca * d1 * factor
        matrix_derivatives[p, 2] = 0.5 * ca * d0 * factor
        matrix_derivatives[p, 3] = -0.5 * sa * d1 * factor

    for p in range(3):
        factor = 0.4 * (1.0 - h * h) * filter_features[p]
        dd0 = -np.sin(beta0) * factor
        dd1 = np.sin(beta1) * factor
        matrix_derivatives[p + 3, 0] = ca * dd0
        matrix_derivatives[p + 3, 1] = -sa * dd1
        matrix_derivatives[p + 3, 2] = sa * dd0
        matrix_derivatives[p + 3, 3] = ca * dd1

    derivatives = np.empty((6, 3, 3))
    for p in range(6):
        da, db, dc, dd = matrix_derivatives[p]

        derivatives[p, 0, 0] = 2.0 * a * da
        derivatives[p, 0, 1] = 2.0 * (da * b + a * db)
        derivatives[p, 0, 2] = 2.0 * b * db

        derivatives[p, 1, 0] = da * c + a * dc
        derivatives[p, 1, 1] = da * d + a * dd + db * c + b * dc
        derivatives[p, 1, 2] = db * d + b * dd

        derivatives[p, 2, 0] = 2.0 * c * dc
        derivatives[p, 2, 1] = 2.0 * (dc * d + c * dd)
        derivatives[p, 2, 2] = 2.0 * d * dd

    return matrix, transfer, derivatives


@njit
def row_likelihood(parameters, features, target):
    n = len(features)
    k = len(target)

    x = np.cos(INITIAL_BOND_ANGLE)
    y = np.sin(INITIAL_BOND_ANGLE)
    initial = np.array([x * x, x * y, y * y])

    sums = np.zeros((k + 1, 3))
    sums[0] = initial
    sum_derivatives = np.zeros((k + 1, 6, 3))

    numerator_state = initial.copy()
    numerator_derivatives = np.zeros((6, 3))
    target_position = 0

    for i in range(n):
        _, transfer, derivatives = local_transfer(parameters, features[i])

        # Opadajuci r cuva prethodno stanje dinamickog programa.
        for r in range(min(k, i + 1), 0, -1):
            for p in range(6):
                for a in range(3):
                    value = 0.0
                    for b in range(3):
                        value += (
                            derivatives[p, a, b] * sums[r - 1, b]
                            + transfer[a, b] * sum_derivatives[r - 1, p, b]
                        )
                    sum_derivatives[r, p, a] += value

            for a in range(3):
                value = 0.0
                for b in range(3):
                    value += transfer[a, b] * sums[r - 1, b]
                sums[r, a] += value

        if target_position < k and i == target[target_position]:
            old_state = numerator_state.copy()
            old_derivatives = numerator_derivatives.copy()

            for a in range(3):
                value = 0.0
                for b in range(3):
                    value += transfer[a, b] * old_state[b]
                numerator_state[a] = value

            for p in range(6):
                for a in range(3):
                    value = 0.0
                    for b in range(3):
                        value += (
                            derivatives[p, a, b] * old_state[b]
                            + transfer[a, b] * old_derivatives[p, b]
                        )
                    numerator_derivatives[p, a] = value

            target_position += 1

    numerator = numerator_state[0] + numerator_state[2]
    normalizer = sums[k, 0] + sums[k, 2]

    if numerator <= 0.0 or normalizer <= 0.0 or target_position != k:
        raise ValueError("Neispravna zajednicka amplituda ili normalizator.")

    loss = np.log(normalizer) - np.log(numerator)
    gradient = np.empty(6)

    for p in range(6):
        gradient[p] = (
            (sum_derivatives[k, p, 0] + sum_derivatives[k, p, 2])
            / normalizer
            - (numerator_derivatives[p, 0] + numerator_derivatives[p, 2])
            / numerator
        )

    return loss, gradient


@njit(parallel=True, nogil=True)
def likelihood_kernel(parameters, features, targets):
    count = len(targets)
    losses = np.empty(count)
    blocks = 16
    partial = np.zeros((blocks, 6))

    for block in prange(blocks):
        start = count * block // blocks
        stop = count * (block + 1) // blocks

        for t in range(start, stop):
            loss, gradient = row_likelihood(
                parameters, features[t], targets[t]
            )
            losses[t] = loss
            for p in range(6):
                partial[block, p] += gradient[p]

    gradient = np.zeros(6)
    for block in range(blocks):
        for p in range(6):
            gradient[p] += partial[block, p] / count

    return losses, gradient


def objective(parameters, features, targets):
    losses, gradient = likelihood_kernel(parameters, features, targets)
    penalty = (
        0.5 * REGULARIZATION * np.dot(parameters, parameters)
        / len(parameters)
    )
    gradient += REGULARIZATION * parameters / len(parameters)
    return float(losses.mean() + penalty), gradient


def fit_model(features, targets, start=None):
    parameters = (
        np.array([0.50, 0.25, -0.20, 0.15, 0.35, -0.25])
        if start is None else np.asarray(start, dtype=np.float64).copy()
    )
    iteration = 0
    last_message = perf_counter()

    def callback(current):
        nonlocal iteration, last_message
        iteration += 1
        if perf_counter() - last_message >= 30.0:
            print("Ucenje: iteracija", iteration, flush=True)
            last_message = perf_counter()

    for attempt in range(2):
        result = minimize(
            objective,
            parameters,
            args=(features, targets),
            method="BFGS",
            jac=True,
            callback=callback,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
            },
        )
        parameters = np.asarray(result.x, dtype=np.float64)
        value, gradient = objective(parameters, features, targets)
        norm = float(np.max(np.abs(gradient)))

        if (
            np.isfinite(value)
            and np.all(np.isfinite(parameters))
            and np.all(np.isfinite(gradient))
            and norm <= GRADIENT_TOLERANCE
        ):
            return parameters

        if attempt == 0:
            print(
                f"Nastavak optimizacije: gradijent={norm:.6g}.",
                flush=True,
            )

    raise RuntimeError(
        f"Optimizacija nije konvergirala; gradijent={norm:.6g}; "
        + str(result.message)
    )


def mean_nll(parameters, features, targets):
    losses, _ = likelihood_kernel(parameters, features, targets)
    result = float(losses.mean())
    if not np.isfinite(result):
        raise RuntimeError("NLL nije konacan.")
    return result


@njit
def matrices_and_normalizer(parameters, features, k):
    n = len(features)
    matrices = np.empty((n, 2, 2))
    sums = np.zeros((k + 1, 3))

    x = np.cos(INITIAL_BOND_ANGLE)
    y = np.sin(INITIAL_BOND_ANGLE)
    sums[0] = np.array([x * x, x * y, y * y])

    for i in range(n):
        matrix, transfer, _ = local_transfer(parameters, features[i])
        matrices[i] = matrix

        for r in range(min(k, i + 1), 0, -1):
            for a in range(3):
                value = 0.0
                for b in range(3):
                    value += transfer[a, b] * sums[r - 1, b]
                sums[r, a] += value

    normalizer = sums[k, 0] + sums[k, 2]
    return matrices, normalizer


@njit
def norm_bounds(matrices, k):
    n = len(matrices)
    bounds = np.zeros((n + 1, k + 1))
    bounds[:, 0] = 1.0

    for i in range(n - 1, -1, -1):
        # Kolone RY(alpha) D su ortogonalne.
        # Maksimalni kvadrat njihove norme je ||M_i||_2^2.
        first = matrices[i, 0, 0] ** 2 + matrices[i, 1, 0] ** 2
        second = matrices[i, 0, 1] ** 2 + matrices[i, 1, 1] ** 2
        maximum = max(first, second)

        for r in range(1, k + 1):
            bounds[i, r] = max(
                bounds[i + 1, r],
                maximum * bounds[i + 1, r - 1],
            )

    return bounds


@njit
def has_five_consecutive(mask):
    return (
        mask
        & (mask >> 1)
        & (mask >> 2)
        & (mask >> 3)
        & (mask >> 4)
    ) != 0


@njit
def search(
    matrices, bounds, position, remaining, v0, v1, mask,
    best_score, best_mask, tolerance
):
    n = len(matrices)
    if n - position < remaining:
        return

    upper = (v0 * v0 + v1 * v1) * bounds[position, remaining]
    if best_score[0] > 0.0:
        if upper <= best_score[0] * (1.0 + tolerance):
            return

    if remaining == 0:
        if not has_five_consecutive(mask):
            value = v0 * v0 + v1 * v1
            if value > best_score[0]:
                best_score[0] = value
                best_mask[0] = mask
        return

    if position >= n:
        return

    m = matrices[position]
    take0 = m[0, 0] * v0 + m[0, 1] * v1
    take1 = m[1, 0] * v0 + m[1, 1] * v1
    take_mask = mask | (np.int64(1) << position)

    take_upper = (
        (take0 * take0 + take1 * take1)
        * bounds[position + 1, remaining - 1]
    )
    skip_upper = (
        (v0 * v0 + v1 * v1)
        * bounds[position + 1, remaining]
    )

    if take_upper >= skip_upper:
        if not has_five_consecutive(take_mask):
            search(
                matrices, bounds, position + 1, remaining - 1,
                take0, take1, take_mask,
                best_score, best_mask, tolerance,
            )
        search(
            matrices, bounds, position + 1, remaining,
            v0, v1, mask, best_score, best_mask, tolerance,
        )
    else:
        search(
            matrices, bounds, position + 1, remaining,
            v0, v1, mask, best_score, best_mask, tolerance,
        )
        if not has_five_consecutive(take_mask):
            search(
                matrices, bounds, position + 1, remaining - 1,
                take0, take1, take_mask,
                best_score, best_mask, tolerance,
            )


def combination_weight(matrices, combination):
    vector = np.array([
        np.cos(INITIAL_BOND_ANGLE),
        np.sin(INITIAL_BOND_ANGLE),
    ])
    for value in combination:
        vector = matrices[value] @ vector
    return float(vector @ vector)


def decode(parameters, features, k=K, tolerance=SEARCH_TOLERANCE):
    matrices, normalizer = matrices_and_normalizer(parameters, features, k)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("Neispravan zavrsni normalizator.")

    bounds = norm_bounds(matrices, k)
    best_score = np.array([-1.0])
    best_mask = np.zeros(1, dtype=np.int64)

    search(
        matrices, bounds, 0, k,
        float(np.cos(INITIAL_BOND_ANGLE)),
        float(np.sin(INITIAL_BOND_ANGLE)),
        np.int64(0), best_score, best_mask, tolerance,
    )

    mask = int(best_mask[0])
    indices = [
        i for i in range(len(features)) if mask & (1 << i)
    ]
    if len(indices) != k or has_five_consecutive(np.int64(mask)):
        raise RuntimeError("Dekoder je vratio neispravnu kombinaciju.")

    independently_scored = combination_weight(matrices, indices)
    if not np.isclose(
        independently_scored, best_score[0], rtol=1e-11, atol=1e-14
    ):
        raise RuntimeError("Nezavisna provera maksimuma nije prosla.")

    return [i + 1 for i in indices], best_score[0] / normalizer


def verify_quantum_operations():
    u = np.ones(N) / np.sqrt(N)
    coefficient = np.exp(-1j * COLLECTIVE_ANGLE) - 1.0
    collective = np.eye(N, dtype=np.complex128) + coefficient * np.outer(u, u)

    embedded = np.eye(64, dtype=np.complex128)
    embedded[1:N + 1, 1:N + 1] = collective

    values = np.arange(1, N + 1)
    vector = np.exp(0.17j * values) * (1.0 + 0.2 * np.cos(values))
    vector /= np.linalg.norm(vector)

    present = np.zeros(N)
    present[[1, 5, 9, 14, 22, 30, 37]] = 1.0
    strength = 2.1

    initial = np.zeros(64, dtype=np.complex128)
    initial[1:N + 1] = vector
    phases = np.ones(64, dtype=np.complex128)
    phases[1:N + 1] = np.exp(
        -1j * strength * (present - present.mean())
    )

    circuit = QuantumCircuit(6)
    circuit.append(StatePreparation(initial), list(range(6)))
    circuit.unitary(embedded, list(range(6)))
    circuit.append(DiagonalGate(phases), list(range(6)))

    state = Statevector.from_instruction(circuit).data[1:N + 1]
    actual = np.outer(state, state.conj())
    expected = unitary_memory_step(
        np.outer(vector, vector.conj()), present, strength
    )

    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-11):
        raise RuntimeError("Qiskit provera memorijske operacije nije prosla.")

    parameters = np.array([0.5, 0.3, -0.2, 0.1, 0.4, -0.3])
    feature = np.array([1.0, 0.25, -0.4, 0.3])
    matrix, _, _ = local_transfer(parameters, feature)

    alpha = parameters[0] + parameters[1] * feature[1] + parameters[2] * feature[2]
    h = np.tanh(
        parameters[3] + parameters[4] * feature[1] + parameters[5] * feature[3]
    )
    beta0 = 0.7 + 0.4 * h
    beta1 = 0.7 - 0.4 * h

    circuit = QuantumCircuit(2)
    # q0: memorijski qubit; q1: pomocni qubit.
    circuit.x(0)
    circuit.cry(float(2.0 * beta0), 0, 1)
    circuit.x(0)
    circuit.cry(float(2.0 * beta1), 0, 1)
    circuit.ry(float(alpha), 0)

    block = Operator(circuit).data[np.ix_([0, 1], [0, 1])]
    if not np.allclose(block, matrix, rtol=1e-10, atol=1e-11):
        raise RuntimeError("Qiskit provera zajednicke amplitude nije prosla.")


def internal_checks():
    parameters = np.array([0.5, 0.25, -0.2, 0.15, 0.35, -0.25])
    count = 5
    n = 9
    features = np.empty((count, n, 4))

    for t in range(count):
        for i in range(n):
            features[t, i] = [
                1.0,
                np.sin(0.3 * (i + 1) + t),
                np.cos(0.4 * (i + 1) - 0.2 * t),
                np.sin(0.7 * i + 0.5 * t),
            ]

    targets = np.array(
        list(combinations(range(n), 3))[::9][:count],
        dtype=np.int64,
    )

    for point in (
        parameters,
        np.array([-0.7, 0.6, 0.3, -0.4, 0.2, 0.5]),
    ):
        _, analytic = objective(point, features, targets)
        numerical = np.empty(6)

        for j in range(6):
            plus = point.copy()
            minus = point.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical[j] = (
                objective(plus, features, targets)[0]
                - objective(minus, features, targets)[0]
            ) / 2e-5

        error = np.max(
            np.abs(analytic - numerical)
            / np.maximum(1.0, np.abs(numerical))
        )
        if error > 2e-5 or not np.all(np.isfinite(analytic)):
            raise RuntimeError("Analiticki gradijent nije prosao proveru.")

    for k in (3, 5):
        matrices, normalizer = matrices_and_normalizer(
            parameters, features[0], k
        )
        total = 0.0
        brute_best = -1.0

        for combination in combinations(range(n), k):
            weight = combination_weight(matrices, combination)
            total += weight
            mask = sum(1 << i for i in combination)
            if not has_five_consecutive(np.int64(mask)):
                brute_best = max(brute_best, weight / normalizer)

            loss, _ = row_likelihood(
                parameters, features[0], np.array(combination, dtype=np.int64)
            )
            if not np.isclose(
                np.exp(-loss), weight / normalizer,
                rtol=1e-10, atol=1e-12,
            ):
                raise RuntimeError("Likelihood i amplituda se ne poklapaju.")

        if not np.isclose(total, normalizer, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Normalizator nije prosao iscrpnu proveru.")

        _, best = decode(parameters, features[0], k=k, tolerance=0.0)
        if not np.isclose(best, brute_best, rtol=1e-10, atol=1e-12):
            raise RuntimeError("Dekoder nije prosao iscrpnu proveru.")

    identical = np.tile(np.array([1.0, 0.2, -0.1, 0.3]), (N, 1))
    example = np.array([0, 3, 8, 14, 21, 30, 38], dtype=np.int64)
    loss, _ = row_likelihood(parameters, identical, example)

    if not np.isclose(loss, np.log(float(comb(N, K))), atol=1e-9):
        raise RuntimeError("Jednaki ulazi stvaraju pristrasnost oznaka.")

    synthetic = np.array([
        sorted({((5 * t + 3 * j) % N) + 1 for j in range(K)})
        for t in range(20)
    ], dtype=np.int64)

    for _, strength, logistic in CONFIGURATIONS:
        full = make_history(synthetic, strength, logistic)
        prefix = make_history(synthetic[:12], strength, logistic)
        if not np.allclose(full[0][:11], prefix[0], atol=1e-12):
            raise RuntimeError("Buduci podaci menjaju ranije karakteristike.")

    verify_quantum_operations()


def process_csv(path):
    print("CSV:", path.name, flush=True)
    draws = load_csv(path)
    count = len(draws) - 1
    train_end = int(0.70 * count)
    validation_end = int(0.85 * count)

    candidates = []
    failures = []

    for name, strength, logistic in CONFIGURATIONS:
        print("Priprema dinamike:", name, flush=True)
        features, targets, final_features = make_history(
            draws, strength, logistic
        )
        print("Ucenje:", name, flush=True)

        try:
            parameters = fit_model(
                features[:train_end], targets[:train_end]
            )
            validation = mean_nll(
                parameters,
                features[train_end:validation_end],
                targets[train_end:validation_end],
            )
            print(f"Validacija NLL: {validation:.6f} | {name}")

            candidates.append({
                "name": name,
                "features": features,
                "targets": targets,
                "final_features": final_features,
                "validation": validation,
                "parameters": parameters,
            })
        except RuntimeError as exc:
            failures.append(name)
            print("PROVERA: kandidat odbacen:", name, "|", str(exc))

    if not candidates:
        raise RuntimeError("Nijedna varijanta nije prosla provere.")

    chosen = min(candidates, key=lambda item: item["validation"])
    print("Izbor validacije:", chosen["name"])

    # Test poredi varijante. Ne bira pobednika ponovo.
    for candidate in candidates:
        print("Ucenje za test:", candidate["name"], flush=True)
        try:
            parameters = fit_model(
                candidate["features"][:validation_end],
                candidate["targets"][:validation_end],
                start=candidate["parameters"],
            )
            score = mean_nll(
                parameters,
                candidate["features"][validation_end:],
                candidate["targets"][validation_end:],
            )
            candidate["test_parameters"] = parameters
            candidate["test_nll"] = score
            print(f"Test NLL: {score:.6f} | {candidate['name']}")
        except RuntimeError as exc:
            print("PROVERA: test model nije prosao:", str(exc))
            if candidate is chosen:
                raise

    if "test_parameters" not in chosen:
        raise RuntimeError("Izabrani model nije zavrsio test proveru.")

    print(
        "Zavrsno ucenje na svim istorijskim prelazima:",
        chosen["name"],
        flush=True,
    )
    final_parameters = fit_model(
        chosen["features"],
        chosen["targets"],
        start=chosen["test_parameters"],
    )

    print("NEXT: deterministicka pretraga zajednicke amplitude.", flush=True)
    prediction, probability = decode(
        final_parameters, chosen["final_features"]
    )
    print("NEXT:", prediction)

    if DETAILS:
        print(f"Verovatnoca cele sedmorke: {probability:.12g}")
        print("Parametri:", final_parameters.tolist())

    return {
        "name": path.name,
        "draws": draws,
        "variant": chosen["name"],
        "next": tuple(prediction),
        "parameters": final_parameters,
        "context": chosen["final_features"],
    }


def main():
    print("Quantum Driven Dynamics V1: memorija, udari i logisticka modulacija.")
    print("Zajednicke kvantne amplitude; tacna normalizacija cele sedmorke.")
    print("Lokalna simulacija; klasicna optimizacija i pretraga.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()
    print("Interne provere: prosle.", flush=True)

    results = []
    failures = 0

    for path in FILES:
        print()
        try:
            results.append(process_csv(path))
        except KeyboardInterrupt:
            print()
            print("Obrada prekinuta korisnickim zahtevom.")
            return 130
        except Exception as exc:
            failures += 1
            print("STOP za ovaj CSV:", str(exc))

    for i in range(len(results)):
        for j in range(i):
            a = results[j]
            b = results[i]
            if a["next"] != b["next"]:
                continue

            same_input = (
                a["draws"].shape == b["draws"].shape
                and np.array_equal(a["draws"], b["draws"])
            )
            print()
            print("PROVERA: isti NEXT za", a["name"], "i", b["name"])
            print(
                f"Isti CSV sadrzaj: {same_input} "
                f"| razlika parametara: "
                f"{np.linalg.norm(a['parameters'] - b['parameters']):.6g} "
                f"| razlika konteksta: "
                f"{np.linalg.norm(a['context'] - b['context']):.6g}"
            )

    if failures:
        print()
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")
        sys.exit(130)
    except Exception as exc:
        print("STOP:", str(exc))
        sys.exit(1)



"""
Quantum Driven Dynamics V1: memorija, udari i logisticka modulacija.
Zajednicke kvantne amplitude; tacna normalizacija cele sedmorke.
Lokalna simulacija; klasicna optimizacija i pretraga.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.



CSV: loto7_4724_k93.csv
Priprema dinamike: memorija
Ucenje: memorija
Validacija NLL: 16.548393 | memorija
Priprema dinamike: fazni udari
Ucenje: fazni udari
Validacija NLL: 16.547404 | fazni udari
Priprema dinamike: logisticka modulacija
Ucenje: logisticka modulacija
Validacija NLL: 16.547467 | logisticka modulacija
Izbor validacije: fazni udari
Ucenje za test: memorija
Test NLL: 16.547579 | memorija
Ucenje za test: fazni udari
Test NLL: 16.551047 | fazni udari
Ucenje za test: logisticka modulacija
Test NLL: 16.550829 | logisticka modulacija
Zavrsno ucenje na svim istorijskim prelazima: fazni udari
NEXT: deterministicka pretraga zajednicke amplitude.
NEXT: [21, x, 29, y, 34, z, 39]



CSV: loto7_4724_k93_loto_2984.csv
Priprema dinamike: memorija
Ucenje: memorija
Validacija NLL: 16.550101 | memorija
Priprema dinamike: fazni udari
Ucenje: fazni udari
Validacija NLL: 16.550385 | fazni udari
Priprema dinamike: logisticka modulacija
Ucenje: logisticka modulacija
Validacija NLL: 16.550884 | logisticka modulacija
Izbor validacije: memorija
Ucenje za test: memorija
Test NLL: 16.551707 | memorija
Ucenje za test: fazni udari
Test NLL: 16.550566 | fazni udari
Ucenje za test: logisticka modulacija
Test NLL: 16.550701 | logisticka modulacija
Zavrsno ucenje na svim istorijskim prelazima: memorija
NEXT: deterministicka pretraga zajednicke amplitude.
NEXT: [3, x, 13, y, 23, z, 28]



CSV: loto7_4724_k93_loto_plus_1740.csv
Priprema dinamike: memorija
Ucenje: memorija
Validacija NLL: 16.551816 | memorija
Priprema dinamike: fazni udari
Ucenje: fazni udari
Validacija NLL: 16.559315 | fazni udari
Priprema dinamike: logisticka modulacija
Ucenje: logisticka modulacija
Validacija NLL: 16.560310 | logisticka modulacija
Izbor validacije: memorija
Ucenje za test: memorija
Test NLL: 16.557863 | memorija
Ucenje za test: fazni udari
Test NLL: 16.555592 | fazni udari
Ucenje za test: logisticka modulacija
Test NLL: 16.554904 | logisticka modulacija
Zavrsno ucenje na svim istorijskim prelazima: memorija
NEXT: deterministicka pretraga zajednicke amplitude.
NEXT: [11, x, 19, y, 21, z, 28]
"""



"""
Nelinearna dinamika sa kontrolisanim zadržavanjem istorije.

Kvantni haos — BlueQubit zadatak procenjuje OTOC korelatore, koji prate širenje uticaja perturbacije kroz kvantni sistem. 
Za NEXT korisna adaptacija je model čije stanje pamti prethodna kola, dok naučene interakcije određuju kako se njihov uticaj dalje razvija. 
Kvantna dinamika već ima istraživačku osnovu za obradu vremenskih podataka. 

Kvantni dinamički modela sa učenom memorijom i jačinom interakcija. 

V1 poredi tri varijante: kvantnu memoriju, memoriju sa faznim udarima i udare sa logističkom modulacijom. 
Validacija bira varijantu za NEXT. 
Zajednička amplituda sedmorke koristi pomoćni qubit koji povezuje svih sedam izbora, uz tačnu normalizaciju cele raspodele.

Poredi kvantnu memoriju, fazne udare i logističku modulaciju. 
Varijantu bira hronološka validacija. 
Sedmorka ima zajedničku amplitudu i tačno izračunat normalizator.

Potrebni su NumPy, SciPy, Numba i Qiskit. Kod nisam izvršavao;
"""
