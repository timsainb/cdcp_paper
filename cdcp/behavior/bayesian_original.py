import numpy as np
from lmfit import Model
import lmfit


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def bayesian_model_new(params, x_true, prior_probability, decision_boundary):
    """
    gamma: the side biases of the bird
    sigma: sigma of the likelihood gaussian
    
    alpha: the proportion of the both cue and categorical stimuli are attended
    beta: the proportion of the time that only categorical is attended
    delta: the proportion of the time that only cue is attended
    kappa: the proportion of the both cue and categorical stimuli are ignored
    
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

    def norm(x):
        return np.array([i / np.sum(i) for i in x])

    strategy_denom = (
        params["alpha_integration"]
        + params["beta_categorical_only"]
        + params["delta_cue_only"]
        + params["kappa_no_attention"]
    )
    alpha_integration = params["alpha_integration"] / strategy_denom
    beta_categorical_only = params["beta_categorical_only"] / strategy_denom
    delta_cue_only = params["delta_cue_only"] / strategy_denom
    kappa_no_attention = params["kappa_no_attention"] / strategy_denom

    likelihood = np.array(
        [gaussian(x_true, x_i, params["sigma_likelihood"]) for x_i in x_true]
    )
    likelihood = norm(likelihood)  # normalize likelihood gaussian to sum to 1

    posterior_probability = norm(prior_probability * likelihood)

    def decision_probaiblity(posterior):
        return np.sum([i * decision_boundary for i in posterior], axis=1)

    decision = (
        decision_probaiblity(posterior_probability) * alpha_integration
        + decision_probaiblity(likelihood) * beta_categorical_only
        + decision_probaiblity(norm(prior_probability * np.ones(np.shape(likelihood))))
        * delta_cue_only
        + decision_probaiblity(
            norm(params["gamma_side_bias"] * np.ones(np.shape(posterior_probability)))
        )
        * kappa_no_attention
    )
    # decision = decision_probaiblity(posterior_probability)

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
    p.add("sigma_likelihood", 10, min=1e-10, max=100.0)
    p.add("alpha_integration", 0.1, min=1e-10, max=1.0)
    p.add("beta_categorical_only", 0.1, min=1e-10, max=1.0)
    p.add("delta_cue_only", 0.1, min=1e-10, max=1.0)
    p.add("kappa_no_attention", 0.1, min=1e-10, max=1.0)
    p.add("gamma_side_bias", 0.5, min=1e-10, max=1.0)  # side bias

    # minimization
    model_minimizer = lmfit.Minimizer(
        bayesian_model_residuals,
        p,
        fcn_args=(x_true, prior, responses, positions, decision_boundary, verbose),
        nan_policy="omit",
    )
    results_model = model_minimizer.minimize(method="nelder")
    # print(bird, cue, condition_type, prior[:1], responses[:3], positions[:3],results_model.params)
    # model_minimizer = lmfit.Minimizer(lnprob, p, fcn_args = (x_true, prior, responses,positions, decision_boundary, verbose))
    # mi.params.add('noise', value=1, min=0.001, max=2) # add a noise parameter to the minimizer
    # results_model = model_minimizer.emcee(burn=300, steps=1000, thin=20, workers=10, params=mi.params)
    return results_model
