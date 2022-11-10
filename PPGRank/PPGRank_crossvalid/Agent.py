import numpy as np


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y


def total_return(E, gamma):
    # Note that for state S_0 reward is R_1
    # E format is  = [S_0, A_0, R_1]
    # Algorithm has k from 1 to M-t, which is basically k = 0 len of Episode
    tot_return = E[0][2]
    for k in range(1, len(E)):
        tot_return += (gamma ** k) * E[k][2]
    return tot_return


def score(E, theta):
    # Current action chosen --> docID:
    a = E[0][1]
    phi_a = E[0][0][1][a][0]  # Ranking feature vector of action a
    # summation of feature vector of the remaining actions in the episode
    val = 0
    sum_pi = 0
    for k in range(len(E)):
        a_t = E[k][1]
        phi_a_t = E[k][0][1][a_t][0]  # 25 x 1
        x = np.matmul(theta.T, np.array(phi_a_t))
        x = exp_normalize(x)
        pi_a_t = np.exp(x)  # theta = 25 x 1
        sum_pi += pi_a_t
        val += np.round(pi_a_t * phi_a_t, 4)  # 25 x 1

    return np.round(phi_a - np.divide(val, sum_pi), 4)


def score2(E, theta):
    # Current action chosen --> docID:
    a = E[0][1]
    phi_a = E[0][0][1][a][0]  # Ranking feature vector of action a
    # summation of feature vector of the remaining actions in the episode
    val = 0
    for k in range(len(E)):
        a_t = E[k][1]
        phi_a_t = E[k][0][1][a_t][0]  # 25 x 1
        pi_a_t = np.matmul(theta.T, np.array(phi_a_t))  # theta = 25 x 1
        val += np.round(pi_a_t * phi_a_t, 4)  # 25 x 1
    return np.round(phi_a - val, 4)
