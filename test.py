import timeit

import pylab 

import numpy as np
from sklearn.linear_model import lars_path, lasso_path, ElasticNet
from sklearn.datasets import load_digits

data = load_digits()
x = data['data']
y = data['target']
# x = np.random.rand(1000, 100)
# y = np.random.rand(1000)

# Our goal is to compute lasso coefs for all alphas given x and y

## First way to do so
def use_lasso_path(x, y, alphas):
    # For some reason, the list of models returned by lasso_path is in
    # decreasing order instead increasing order, so we reverse it.
    tmp = lasso_path(x, y, alphas=alphas, copy_X=True)
    tmp.reverse()
    return tmp

## Second way to do so
# Here, we use lars_path to compute the kinks (hitting points) of
# Lasso and interpolate by hand for each alpha in alphas.
def use_lars_path(x, y, alphas):
    kinks, _, coefs = lars_path(x, y, method='lasso', copy_X=True)
    # kinks are given in decreasing order and we want to go through
    # them in increasing order.
    kinks = kinks[::-1]
    n_kinks = len(kinks)

    # Since we iterate over the kinks in reverse order, we have to
    # reverse the coefs array too
    coefs = coefs[:,::-1]
    coefs_ = {}

    # Since we iterate on alphas and on kinks in increasing order for
    # both, we don't need can always start the next iteration with the
    # last index we've found.
    # r: iteration index
    r = 0
    for alpha in alphas:
        # Find the two kinks between which alpha lies
        while r != n_kinks - 1 and not kinks[r] <= alpha <= kinks[r + 1]:
            r += 1

        # There's no such two kinks (ie alpha > max(kinks))
        if r == n_kinks - 1:
            # In that case the lasso coef are just 0 everywhere
            coefs_[alpha] = np.zeros(coefs.shape[0])
            continue

        # We do the linear interpolation between the two kinks
        x_ = (alpha - kinks[r]) / (kinks[r+1] - kinks[r])
        coefs_[alpha] = (1 - x_) * coefs[:, r] + x_ * coefs[:, r+1]
    return coefs_

# Check that the coefs between lasso_path and our interpolation are
# not different.

alphas=np.logspace(-20,20,100)
d = []
lasso = use_lasso_path(x, y, alphas)
coefs__ = use_lars_path(x, y, alphas)
for i in range(len(alphas)):
   d.append(np.abs(np.abs(lasso[i].coef_) - np.abs(coefs__[alphas[i]])))
a = np.array(d)
print 'use_lars_path and lasso_path:', a.mean()

a = a.reshape(-1)
a = np.log10(a)
a[a==-np.inf] = np.inf
a[a==np.inf] = a.min() - 5
close()
subplot(221)
pylab.hist(a)
subplot(222)
pylab.boxplot(a)
suptitle('log10 of absolute difference between actual lasso_path and proposed lasso_path')
    
alphas = np.arange(0, 10000, 1000)
# Let's make some performance measures. It's very hackish.
# dla = []
# dlp = []
# for n in alphas:
#     alphas = np.logspace(-20, 20, num=n)
#     la = timeit.timeit('lasso_path(x, y, alphas=alphas, copy_X=True)',
#                        'from __main__ import lasso_path, x, y, alphas',
#                        number=1)
#     lp = timeit.timeit('use_lars_path(x, y, alphas=alphas)',
#                        'from __main__ import use_lars_path, x, y, alphas',
#                        number=1)
#     print 'alpha:', n, 'speedup:', la/lp
#     dla.append(la)
#     dlp.append(lp)

# dla = np.array(dla)
# dlp = np.array(dlp)

subplot(223)
pylab.plot(alphas, dla, label='Old lasso_path', color='red', linewidth=1.5)
pylab.plot(alphas, dlp, label='Proposed lasso_path', color='black', linewidth=1.5)
legend(loc=0)
xlabel('# alphas to compute')
ylabel('Time (s)')

subplot(224)
pylab.plot(alphas, dla/dlp)
xlabel('# alphas to compute')
ylabel('Speedup')

# TODO

# intercept: y.mean() - np.dot(x.mean(axis=0), coefs__[1e-10].T)

# Let's create the model

#alpha = alphas[1]
#ElasticNet(alpha=alpha, l1_ratio=1,
#           fit_intercept=True, normalize=False)
           
