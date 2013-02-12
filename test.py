import timeit

import numpy as np
from sklearn.linear_model import lars_path, lasso_path, ElasticNet
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
x = diabetes['data']
y = diabetes['target']
# x = np.random.rand(1000, 100)
# y = np.random.rand(1000)

# Our goal is to compute lasso coefs for all alphas.

## First way to do so
def use_lasso_path(x, y, alphas):
    tmp = lasso_path(x, y, alphas=alphas, copy_X=True)
    tmp.reverse()
    return tmp

## Second way to do so
# Here, we use lars_path to compute the kinks (hitting points) of
# Lasso and manually interpolate for each alpha in alphas.
def use_lars_path(x, y, alphas):
    kinks, _, coefs = lars_path(x, y, method='lasso', copy_X=True)
    # kinks are given in decreasing order and we want to go through
    # them in increasing order.
    kinks = kinks[::-1]
    nkinks = len(kinks)

    # Since we iterate over the kinks in reverse order, we have to
    # reverse the coefs array too
    coefs = coefs[:,::-1]
    coefs_ = {}

    r = 0
    for alpha in alphas:
        while r != nkinks - 1 and not kinks[r] <= alpha <= kinks[r + 1]:
            r += 1
            if r == nkinks - 1: break
        if r == nkinks - 1:
            coefs_[alpha] = np.zeros(coefs.shape[0])
            continue
        x_ = (alpha - kinks[r]) / (kinks[r+1] - kinks[r])
        coefs_[alpha] = (1 - x_) * coefs[:, r] + x_ * coefs[:, r+1]
    return coefs_

# Check that the coefs between lasso_path and our interpolation are
# not different.

# For some reason, the list of models returned by lasso_path is in
# decreasing order instead increasing order, so we reverse it.

alphas=np.logspace(-20,20,100)
d = []
lasso = use_lasso_path(x, y, alphas)
coefs__ = use_lars_path(x, y, alphas)
for i in range(len(alphas)):
   d.append(np.abs(np.abs(lasso[i].coef_) - np.abs(coefs__[alphas[i]])))
a = np.array(d)
print 'use_lars_path and lasso_path:', a.mean()

# Let's make some performance measures. It's very hackish.
# dla = []
# dlp = []
# for n in np.arange(0, 1000, 50):
#     alphas = np.logspace(-20, 20, num=n)
#     la = timeit.timeit('lasso_path(x, y, alphas=alphas, copy_X=True)',
#                        'from __main__ import lasso_path, x, y, alphas',
#                        number=1)
#     lp = timeit.timeit('use_lars_path(x, y, alphas=alphas)',
#                        'from __main__ import use_lars_path, x, y, alphas',
#                        number=1)
#     print n, la, lp
#     dla.append(la)
#     dlp.append(lp)

# intercept: y.mean() - np.dot(x.mean(axis=0), coefs__[1e-10].T)

# Let's create the model

#alpha = alphas[1]
#ElasticNet(alpha=alpha, l1_ratio=1,
#           fit_intercept=True, normalize=False)
           
