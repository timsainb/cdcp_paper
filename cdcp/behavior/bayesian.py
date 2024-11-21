import numpy as np
from lmfit import Model
import lmfit


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def bayesian_model(params, x_true, prior_probability, decision_boundary):
    """
    gamma: the side biases of the bird
    sigma: sigma of the likelihood gaussian
    delta: the overall innattentivity to cue stimuli
    alpha: the overall innattentivity all stimuli 
    beta: the overall innattentivity to the categorical stimuli
    Parameters
    ----------
    params : [type]
        [description]
    x_true : [type]
        [description]
    prior_probability : [type]
        [description]
    decision_boundary : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    side_bias = (decision_boundary * (1 - ((1 - params["gamma_side_bias"]) * 2))) + (
        1 - params["gamma_side_bias"]
    )
    side_bias = side_bias / np.sum(side_bias)  # normalize the side bias
    # the likelihood is list of gaussians centered around x_true (the interpolation point) at for each value of x_true
    likelihood = np.array(
        [gaussian(x_true, x_i, params["sigma_likelihood"]) for x_i in x_true]
    )
    likelihood = np.array(
        [i / np.sum(i) for i in likelihood]
    )  # normalize likelihood gaussian to sum to 1

    likelihood = np.array(
        [
            (1 - params["beta_categorical_attention"]) * i
            + params["beta_categorical_attention"] * side_bias
            for i in likelihood
        ]
    )
    likelihood = np.array(
        [i / np.sum(i) for i in likelihood]
    )  # normalize full likelihood to sum to 1

    # calculate the bias of the prior probability
    prior_probability = (
        1.0 - params["delta_cue_attention"]
    ) * prior_probability + side_bias * params["delta_cue_attention"]

    # calculate the posterior probability
    posterior_probability = likelihood * prior_probability
    posterior_probability = np.array(
        [i / np.sum(i) for i in posterior_probability]
    )  # normalize posterior probability to sum to 1
    # posterior_probability = params['alpha']*posterior_probability+((1.-params['alpha'])*params['gamma'])
    posterior_probability = (
        1.0 - params["alpha_overall_attention"]
    ) * posterior_probability + ((params["alpha_overall_attention"]) * side_bias)
    posterior_probability = np.array(
        [i / np.sum(i) for i in posterior_probability]
    )  # normalize posterior probability to sum to 1

    # make a decision
    decision = np.sum([i * decision_boundary for i in posterior_probability], axis=1)
    return decision, posterior_probability, likelihood


def side_bias(gamma, decision_boundary):
    return (decision_boundary * (1 - (1 - gamma) * 2)) + (1 - gamma)


def bayesian_model_residuals(
    params,
    x_true,
    prior_probability,
    responses,
    positions,
    decision_boundary,
    verbose=False,
):
    decision, _, _ = bayesian_model(
        params, x_true, prior_probability, decision_boundary
    )
    residuals = responses - [decision[x_true == i][0] for i in positions]
    if verbose:
        print(np.sum(residuals ** 2))
    return residuals


def lnprob(
    params,
    x_true,
    prior_probability,
    responses,
    positions,
    decision_boundary,
    verbose=False,
):
    """
    this isn't fully set up, but we can get the posterior probability of the parameters after fitting
    https://lmfit.github.io/lmfit-py/fitting.html#minimizer-emcee-calculating-the-posterior-probability-distribution-of-parameters
    """
    noise = p["noise"]
    resid = residual(
        p,
        x_true,
        prior_probability,
        responses,
        positions,
        decision_boundary,
        verbose=False,
    )
    return -0.5 * np.sum((resid / noise) ** 2 + np.log(2 * np.pi * noise ** 2))


def fit_bayesian_model(
    bird,
    cue,
    condition_type,
    responses,
    positions,
    x_true,
    prior,
    decision_boundary,
    verbose=False,
):
    """
    responses: bird's behavioral responses
    x_true: the position in the interpolation corresponding to the response 
    """
    if len(responses) == 0:
        return
    # define up parameters passed to model
    p = lmfit.Parameters()
    p.add("sigma_likelihood", 10, min=0, max=100.0)
    p.add(
        "alpha_overall_attention", 0.1, min=0.0, max=1.0
    )  # pct of times ignoring both stimuli
    p.add(
        "beta_categorical_attention", 0.1, min=0.0, max=1.0
    )  # pct of time ignoring the categorical stimulus
    p.add("delta_cue_attention", 0.1, min=0.0, max=1.0)  # pct of time ignoring the cue
    p.add("gamma_side_bias", 0.5, min=0.0, max=1.0)  # side bias

    # minimization
    model_minimizer = lmfit.Minimizer(
        bayesian_model_residuals,
        p,
        fcn_args=(x_true, prior, responses, positions, decision_boundary, verbose),
        nan_policy="omit",
    )
    results_model = model_minimizer.minimize(method="leastsq")
    # print(bird, cue, condition_type, prior[:1], responses[:3], positions[:3],results_model.params)
    # model_minimizer = lmfit.Minimizer(lnprob, p, fcn_args = (x_true, prior, responses,positions, decision_boundary, verbose))
    # mi.params.add('noise', value=1, min=0.001, max=2) # add a noise parameter to the minimizer
    # results_model = model_minimizer.emcee(burn=300, steps=1000, thin=20, workers=10, params=mi.params)
    return results_model

