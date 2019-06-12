import pyDOE

def LatinHyperCubicSampling(minvals,maxvals,num_points,algo_lhs = 'maximin',iterations = 1):
    # Assert minvals and maxvals have the same size:
    assert len(minvals) == len(maxvals)

    # Number of parameters in the space:
    num_params = len(minvals)

    # Generate points, sampled uniformly in [-1, +1] intervals:
    points     = pyDOE.lhs(
        n          = num_params,
        samples    = num_points,
        criterion  = algo_lhs,
        iterations = iterations)
    
    # Set the generated points inside the right intervals
    for i,(minv,maxv) in enumerate(zip(minvals,maxvals)):
        points[:,i] = minv + (maxv-minv) * points[:,i]

    return points