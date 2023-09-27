import numpy as np


def randamax(v, t=None, i=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if i is None:
        idxs = np.where(v == np.amax(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amax(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


def randamin(v, t=None, i=None):
    """
    v: array of values
    t: array used to break ties
    i: array of indices from which we should return an amin
    """
    if i is None:
        idxs = np.where(v == np.amin(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amin(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


def hamiltonian_policy(
        nbr_state,
        nbr_action,
        transition,
        reward,
        gamma=None,
        eps=1e-5,
        max_iter=3000
):
    max_iter = max(nbr_state, max_iter)
    eps = min(1. / np.sqrt(nbr_state), eps)
    ctr = 0
    stop = True
    phi = np.zeros(nbr_state, dtype=float)
    previous_phi = np.zeros(nbr_state, dtype=float)
    hamiltonian_pi = np.zeros(nbr_state, dtype=int)
    while not stop:
        ctr += 1
        for state in range(nbr_state):
            u = - np.inf
            for action in range(nbr_action):
                psa = transition[state, action]
                rsa = reward[state, action]
                if gamma:
                    if rsa + gamma * psa @ phi >= u:
                        u = rsa + gamma * psa @ phi
                        hamiltonian_pi[state] = action
                else:
                    if rsa + psa @ phi >= u:
                        u = rsa + psa @ phi
                        hamiltonian_pi[state] = action
            phi[state] = u
        if gamma is not None:
            phi = phi - np.min(phi)
        delta = np.max(np.abs(phi - previous_phi))
        previous_phi = np.copy(phi)
        stop = (delta < eps) or (ctr > max_iter)
    return hamiltonian_pi


def exp_reward(array, inv_temperature):
    z = np.exp(- inv_temperature * array)
    return z / z.sum()


def linear_reward(array):
    z = array
    return 1. - z / max(z.sum(), 1.)


def pull_precision(nbr_pull, density, harmonic=False):
    if harmonic:
        n = np.sum(density / nbr_pull.astype(float))
        n = 1. / n
    else:
        n = min(nbr_pull[density > 0.])
    return n


def gain_and_density(p, r):
    gain_pool = []
    density_pool = []

    evals, evecs = np.linalg.eig(p.T)
    evec1 = evecs[:, np.isclose(evals, 1)]

    for i in range(evec1.shape[1]):
        evec = evec1[:, i]
        stationary = evec / evec.sum()
        stationary[stationary <= 0.] = 0.
        stationary = stationary / stationary.sum()

        gain = np.sum(stationary * r)

        gain_pool.append(gain)
        density_pool.append(stationary)

    return gain_pool, density_pool


def neighbor_policies(
        nbr_state,
        nbr_action,
        transition,
        reward,
        nbr_pull,
        pi
):
    policy_pool = []
    gain_pool = []
    precision_pool = []
    return_state_pool = []
    mask_pool = []

    for state in range(nbr_state):
        for action in range(nbr_action):
            if action != pi[state]:
                policy = np.copy(pi)
                policy[state] = action
                p = transition[range(nbr_state), policy]
                r = reward[range(nbr_state), policy]
                n = nbr_pull[range(nbr_state), policy]

                gains, densities = gain_and_density(p, r)
                for gain, density in zip(gains, densities):
                    policy_pool.append(policy)
                    gain_pool.append(gain)
                    precision_pool.append(pull_precision(n, density))
                    return_state_pool.append(np.argmax(density))
                    mask_pool.append(density > 0.)

    return policy_pool, gain_pool, precision_pool, return_state_pool, mask_pool


def random_policies(
        nbr_state,
        nbr_action,
        transition,
        reward,
        nbr_pull,
        nbr_policies=10
):
    policy_pool = []
    gain_pool = []
    precision_pool = []
    return_state_pool = []
    mask_pool = []

    for _ in range(nbr_policies):
        policy = np.random.randint(0, nbr_action, nbr_state)
        p = transition[range(nbr_state), policy]
        r = reward[range(nbr_state), policy]
        n = nbr_pull[range(nbr_state), policy]

        gains, densities = gain_and_density(p, r)
        for gain, density in zip(gains, densities):
            policy_pool.append(policy)
            gain_pool.append(gain)
            precision_pool.append(pull_precision(n, density))
            return_state_pool.append(np.argmax(density))
            mask_pool.append(density > 0.)

    return policy_pool, gain_pool, precision_pool, return_state_pool, mask_pool
