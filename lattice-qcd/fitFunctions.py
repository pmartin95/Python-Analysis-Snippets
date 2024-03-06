import numpy as np

def curveFitSigmoidal(x,a,b,c,d):
    return a + b/(1.0 + np.exp(c*(x-d)))

def fitFuncLinear(p,x):
    return p[0]*x + p[1]


def fitSigmoidal(p,x):
    return p[0] + p[1]/(1.0 + np.exp(p[2]*(x-p[3])))