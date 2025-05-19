import jax
import jax.numpy as jnp
from jax.lax import stop_gradient

def sd_1st_cdf(x, y, rel_tau=0.3, get_utility=False):
    """First-order stochastic dominance loss. Approximate 1(x>0) by 

    Args:
        x, y: Scalar array containing samples from two distributions, which we want to maximize.
        rel_tau: Softmax temperature control
        get_utility: Return array u(x) instead of a scalar loss

    Returns:
        Loss value to minimize, or the utility function u(x)
    """    
    nX, nY = len(x), len(y)
    is_y = jnp.concatenate([jnp.zeros_like(x, dtype=float), jnp.ones_like(y, dtype=float)])
    values = jnp.concatenate([x, y])
    idx_sort = jnp.argsort(values)
    sorted_is_y = is_y[idx_sort]
    sorted_values = values[idx_sort]

    F1x = jnp.cumsum(1-sorted_is_y)/nX
    F1y = jnp.cumsum(sorted_is_y)/nY
    eps = jnp.finfo(x.dtype).eps
    
    eta_values = stop_gradient(sorted_values + eps)
    # mu = stop_gradient(F1x > F1y).astype(x.dtype)
    tau = (jnp.max(F1x - F1y) - jnp.min(F1x - F1y))*rel_tau
    mu = jnp.exp(((F1x - F1y) - jnp.max(F1x - F1y))/tau)
    mu = mu/(jnp.sum(mu)+eps)
    mu = stop_gradient(mu).astype(x.dtype)

    eta = jnp.expand_dims(eta_values, axis=1)

    # Previous code (Dai 2023) suggests relu
    if get_utility:
        ux = jnp.sum(jax.nn.relu(eta - jnp.expand_dims(x, axis=0))*jnp.expand_dims(mu, axis=1), axis=0)
        return ux
    else:
        ex = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(x, axis=0)), axis=1)
        # ex = jnp.mean(jax.nn.sigmoid(eta - jnp.expand_dims(x, axis=0)), axis=1)
        loss = jnp.sum(ex*mu)
        return loss

def sd_2nd_cdf(x, y, rel_tau=0.3, get_utility=False):
    """Second-order stochastic dominance loss. Implements algorithm 2

    Args:
        x, y: Scalar array containing samples from two distributions, which we want to maximize.
              x - is the new samples, y - is the reference
              x - X_{\theta_{t,\bar{t}}, y - X_{\theta_t}
        rel_tau: Softmax temperature control
        get_utility: Return array u(x) instead of a scalar loss
        Shicong commented that when working with sampling dependent on theta (Algorithm 3),
        you can pass get_utility=True to get ux values and plug it into the REINFORCE algorithm in place of cumulative rewards.

    Returns:
        Loss value to minimize, or the utility function u(x)
    """    
    nX, nY = len(x), len(y)
    # Single list of 0's and 1's corresponding to x's and y's
    is_y = jnp.concatenate([jnp.zeros_like(x, dtype=float), jnp.ones_like(y, dtype=float)])
    eta = jnp.concatenate([x, y])
    idx_sort = jnp.argsort(eta) # returns indices of eta that would result in it being sorted
    sorted_is_y = is_y[idx_sort] # reshuffle is_y to correspond to the sorted-eta order
    sorted_eta = eta[idx_sort] # sort eta

    # Calculate \hat{F}_1(X) and \hat{F}_1(Y) for all eta
    #F1x = jnp.cumsum(1-sorted_is_y)/nY # 1-sorted_is_y flips the values, s.t. 1's correspond to x's. WHY DIVIDE BY nY AND NOT nX?
    F1x = jnp.cumsum(1-sorted_is_y)/nX # DG: fixed nX

    #F1y = jnp.cumsum(sorted_is_y)/nX # WHY DIVIDE BY nX AND NOT nY?
    F1y = jnp.cumsum(sorted_is_y)/nY # DG: fixed nY
    # Calculate eta_i - eta_{i-1}
    h = sorted_eta - jnp.roll(sorted_eta,1) #jnp.roll is circular shift rigt one place

    F2x_incre = h*jnp.roll(F1x,1) # (eta_i - eta_{i-1})*F_1(X; eta_{i-1}); roll shifts right by one place to align F_1 and h
    F2y_incre = h*jnp.roll(F1y,1) # (eta_i - eta_{i-1})*F_1(Y; eta_{i-1})
    F2x_incre = F2x_incre.at[0].set(0) # zero out the 1st index in-place
    F2y_incre = F2y_incre.at[0].set(0)

    # Calculate F2x for all etas
    F2x = jnp.cumsum(F2x_incre)
    F2y = jnp.cumsum(F2y_incre)

    """
    assuming that x_0 < x_1 < ... < x_n is sorted, for any x_i the corresponding F2x entry 
    before the gradient correction line is computed as F_2(x_i) = \frac{1}{n} \sum_{j=0}^{i-1} (x_i - x_j). 
    (Practically, the division by 1/n is omitted.)
    After this line we cancel out the gradient of x_i, and essentially convert it to 
    F_2(stop_grad(x_i)) = \frac{1}{n} \sum_{j=0}^{i-1} (stop_grad(x_i) - x_j). 
    The reason is that x_i is to be picked up by an argmax/softmax operator mu, 
    and per Danskin's theorem it should be detached from gradient computation.
    softmax is a smooth approximator of argmax
    """
    # correct gradient computation
    F2x = F2x + (1-sorted_is_y)*F1x*(stop_gradient(sorted_eta)-sorted_eta)
    F2y = stop_gradient(F2y)

    eps = jnp.finfo(x.dtype).eps

    tau = (jnp.max(F2x - F2y) - jnp.min(F2x - F2y))*rel_tau
    mu = jnp.exp(((F2x - F2y) - jnp.max(F2x - F2y))/tau)
    mu = mu/(jnp.sum(mu)+eps)
    mu = stop_gradient(mu).astype(x.dtype)

    """
    Calculates the utility in lines 6,7 or returns sum(F2x-F2y)*mu
    which applies when sampling of x_i is independent of theta.
    Note that eventhough we search for the worst input distribution dependent on theta,
    we do not sample from the input distribution - only the weights are adjusted, so
    this trick still applies.
    """
    if get_utility:
        u1 = jnp.cumsum(mu[::-1])[::-1]
        u2_incre = (jnp.roll(sorted_eta,-1) - sorted_eta)*jnp.roll(u1,-1)
        u2_incre = u2_incre.at[-1].set(0)
        u2 = jnp.cumsum(u2_incre[::-1])[::-1]
        ux = u2[jnp.argsort(idx_sort)[:nX]]
        return ux
    else:
        # Create a loss function (of theta) in such a way that it can be differentiated to obtain the gradients
        # w.r.t. theta to improve theta. This is done by using Dankin's theorem.
        loss = jnp.sum((F2x - F2y)*mu)
        return loss

# Straightforward O(N^2) implementation
# def sd_2nd_cdf_(x, y, rel_tau=0.3, get_utility=False):

#     values = jnp.concatenate([x, y])
#     eps = jnp.finfo(x.dtype).eps
    
#     eta = stop_gradient(jnp.expand_dims(values, axis=1))

#     F2x = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(x, axis=0)), axis=1)
#     F2y = jnp.mean(jax.nn.relu(eta - jnp.expand_dims(y, axis=0)), axis=1)

#     tau = (jnp.max(F2x - F2y) - jnp.min(F2x - F2y))*rel_tau
#     mu = jnp.exp(((F2x - F2y) - jnp.max(F2x - F2y))/tau)
#     mu = mu/(jnp.sum(mu)+eps)
#     mu = stop_gradient(mu)

#     if get_utility:
#         ux = jnp.sum(jax.nn.relu(eta - jnp.expand_dims(x, axis=0))*jnp.expand_dims(mu, axis=1), axis=0)
#         return ux
#     else:
#          """
#          If mu were argmax, it'd select from F2x(eta)-F2y(eta) the entry corresponding to the maximal value according to mu.
#          Since it's a softmax approximation, sum() is applied.
#          """
#          loss = jnp.sum((F2x - F2y)*mu)
#          return loss

def mean_risk(x):
    mean = jnp.mean(x)
    return mean + 0.5*jnp.mean(jnp.abs(x-mean))