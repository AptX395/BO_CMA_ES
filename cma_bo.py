from my_bayes_opt import BayesianOptimization
import numpy as np
import my_cma

def rf_cv(Cc,C1,Cmu):
    es = my_cma.CMAEvolutionStrategy(Cc, C1, Cmu, 5 * [0], 0.8, {'seed':400000})
    self = es.optimize(my_cma.ff.rosen)
    val = self.result[1];
    return -val;

rf_bo = BayesianOptimization(
        rf_cv,
        {'Cc': (0.1, 0.999),
        'C1': (0.01, 0.0999),
        'Cmu': (0.01, 0.0999)}
    )

rf_bo.maximize()
