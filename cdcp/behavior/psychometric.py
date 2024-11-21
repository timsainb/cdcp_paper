import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


import lmfit

# fitting model
def fit_model_iter(model, n_iter=10, **kwargs):
    """re-fit model n_iter times and choose the best fit
    chooses method based upon best-fit
    """
    models = []
    AICs = []
    for iter in np.arange(n_iter):
        results_model = model.minimize(**kwargs)
        models.append(results_model)
        AICs.append(results_model.aic)
    return models[np.argmin(AICs)]


# model fit quality
def residuals(y_true, y_model, x, logscaled=False):
    if logscaled:
        return np.abs(np.log(y_true) - np.log(y_model)) * (1 / (np.log(1 + x)))
    else:
        return np.abs(y_true - y_model)


def model_res(p, x, y, model, logscaled=False):
    return residuals(y, model(p, x), x, logscaled=logscaled)


def FourParameterLogistic(p, x):
    """source: https://www.myassays.com/four-parameter-logistic-regression.html
    _min = the minimum value that can be obtained (i.e. what happens at 0 dose)
    slope = Hill’s slope of the curve (i.e. this is related to the steepness of the curve at point c)
    inflection = the point of inflection (i.e. the point on the S shaped curve halfway between a and d)
    _max = the maximum value that can be obtained (i.e. what happens at infinite dose)
    """
    return p["_max"] + (
        (p["_min"] - p["_max"]) / (1 + (x / p["inflection"]) ** p["slope"])
    )


def get_y(model, results, x):
    return model({i: results.params[i].value for i in results.params}, x)


def fit_FourParameterLogistic(
    x,
    y,
    n_iter=1,
    method=["nelder", "leastsq", "least-squares"],
    _min=0.1,
    _max=0.9,
    _inflection=64,
    _slope=10,
    _min_bounds=[1e-10, 1 - 1e-10],
    _max_bounds=[1e-10, 1 - 1e-10],
    _inflection_bounds=[2, 126],
):

    p_logistic = lmfit.Parameters()
    p_logistic.add_many(
        ("_min", _min, True, _min_bounds[0], _min_bounds[1]),
        ("_max", _max, True, _max_bounds[0], _max_bounds[1]),
        ("inflection", _inflection, True, _inflection_bounds[0], _inflection_bounds[1]),
        ("slope", _slope, True, 1e-10, 100),
    )
    results_logistic = lmfit.Minimizer(
        model_res, p_logistic, fcn_args=(x, y, FourParameterLogistic), nan_policy="omit"
    )

    results_logistic = [
        fit_model_iter(results_logistic, n_iter=n_iter, **{"method": meth})
        for meth in method
    ]
    results_logistic = results_logistic[np.argmin([i.aic for i in results_logistic])]

    y_model = get_y(FourParameterLogistic, results_logistic, x)

    residuals = y - y_model
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    _min = results_logistic.params.valuesdict()["_min"]
    _max = results_logistic.params.valuesdict()["_max"]
    _inflection = results_logistic.params.valuesdict()["inflection"]
    _slope = results_logistic.params.valuesdict()["slope"]

    # print(_min, _max, _inflection, _slope, r_squared)

    return (_min, _max, _inflection, _slope), results_logistic, y_model, r_squared


def get_y(model, results, x):
    return model({i: results.params[i].value for i in results.params}, x)


def depricated_FourParameterLogistic(x, a, d, c, b):
    """source: https://www.myassays.com/four-parameter-logistic-regression.html
    a = the minimum value that can be obtained (i.e. what happens at 0 dose)
    b = Hill’s slope of the curve (i.e. this is related to the steepness of the curve at point c)
    c = the point of inflection (i.e. the point on the S shaped curve halfway between a and d)
    d = the maximum value that can be obtained (i.e. what happens at infinite dose)
    """
    return d + ((a - d) / (1 + (x / c) ** b))


def depricated_fit_FourParameterLogistic(x, y):

    # Fit the four parameter logistic
    (_min, _max, _inflection, _slope), (
        _min_cov,
        _max_cov,
        _inflection_cov,
        _slope_cov,
    ) = curve_fit(FourParameterLogistic, x, y, maxfev=10000)
    y_model = FourParameterLogistic(x, _min, _max, _inflection, _slope)

    return (
        (_min, _max, _inflection, _slope),
        (_min_cov, _max_cov, _inflection_cov, _slope_cov),
        y_model,
    )
