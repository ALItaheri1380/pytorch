import random
import math
import numpy as np
import matplotlib.pyplot as plt


def obj(x):
    """
    The sphere function
    :param x:
    :return:
    """
    num = 0
    for i in range(len(x)):
        num += x[i] ** 2
    return num


def boundary_check(x, lb, ub, dim):
    """
    Check the boundary
    :param x: a candidate solution
    :param lb: the lower bound (list)
    :param ub: the upper bound (list)
    :param dim: dimension
    :return:
    """
    for i in range(dim):
        if x[i] < lb[i]:
            x[i] = lb[i]
        elif x[i] > ub[i]:
            x[i] = ub[i]
    return x


def main(ipop, mpop, iter, smin, smax, isigma, fsigma, lb, ub):
    """
    The main function of the IWO
    :param ipop: The initial population size
    :param mpop: The maximum population size
    :param iter: The maximum number of iterations
    :param smin: The minimum number of seeds
    :param smax: The maximum number of seeds
    :param isigma: The initial value of standard deviation
    :param fsigma: The final value of standard deviation
    :param lb: The lower bound (list)
    :param ub: The upper bound (list)
    :return:
    """
    # Step 1. Initialization
    dim = len(lb)  # dimension
    pos = []  # the position of weeds
    score = []  # the score of weeds
    for _ in range(ipop):
        pos.append([random.uniform(lb[i], ub[i]) for i in range(dim)])
        score.append(obj(pos[-1]))
    gbest = min(score)  # the global best
    gbest_pos = pos[score.index(gbest)].copy()  # the global best individual
    iter_best = []  # the global best of each iteration
    con_iter = 0  # the convergence iteration

    # Step 2. The main loop
    for t in range(iter):

        # Step 2.1. Update standard deviation
        sigma = ((iter - t - 1) / (iter - 1)) ** 2 * (isigma - fsigma) + fsigma

        # Step 2.2. Reproduction
        new_pos = []
        new_score = []
        min_score = min(score)
        max_score = max(score)
        for i in range(len(pos)):
            ratio = (score[i] - max_score) / (min_score - max_score)
            snum = math.floor(smin + (smax - smin) * ratio)  # the number of seeds

            for _ in range(snum):
                temp_pos = [pos[i][j] + random.gauss(0, sigma) for j in range(dim)]
                temp_pos = boundary_check(temp_pos, lb, ub, dim)
                new_pos.append(temp_pos)
                new_score.append(obj(temp_pos))

        # Step 2.3. Competitive exclusion
        new_pos.extend(pos)
        new_score.extend(score)

        if len(new_pos) > mpop:
            pos = []
            score = []
            sorted_index = np.argsort(new_score)
            for i in range(mpop):
                pos.append(new_pos[sorted_index[i]])
                score.append(new_score[sorted_index[i]])
        else:
            pos = new_pos
            score = new_score

        # Step 2.4. Update the global best
        if min(score) < gbest:
            gbest = min(score)
            gbest_pos = pos[score.index(gbest)]
            con_iter = t + 1
        iter_best.append(gbest)

    # Step 3. Sort the results
    x = [i for i in range(iter)]
    iter_best = [math.log10(iter_best[i]) for i in range(iter)]
    plt.figure()
    plt.plot(x, iter_best, linewidth=2, color='blue')
    plt.xlabel('Iteration number')
    plt.ylabel('The logarithmic value of gbest')
    plt.title('Convergence curve')
    plt.show()
    return {'best score': gbest, 'best solution': gbest_pos, 'convergence iteration': con_iter}


if __name__ == '__main__':
    ipop = 20
    mpop = 100
    iter = 2000
    smin = 0
    smax = 5
    isigma = 1
    fsigma = 1e-6
    lb = [-10] * 30
    ub = [10] * 30
    print(main(ipop, mpop, iter, smin, smax, isigma, fsigma, lb, ub))