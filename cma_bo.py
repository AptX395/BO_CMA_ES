from my_bayes_opt import BayesianOptimization
import numpy as np
import my_cma



def funcmin(objective_function,x0,sigma):
    def rf_cv(Cc, C1, Cmu):
        es = my_cma.CMAEvolutionStrategy(Cc, C1, Cmu, x0, sigma)
        self = es.optimize(objective_function)
        val = self.result[1];
        return -val;

    rf_bo = BayesianOptimization(
            rf_cv,
            {'Cc': (0.1, 0.999),
            'C1': (0.01, 0.0999),
            'Cmu': (0.01, 0.0999)}
    )

    rf_bo.maximize()
    #print(rf_bo.res['max'])
    # print(rf_bo.res['all'])
    value = rf_bo.res;
    value['max']['max_val'] = -value['max']['max_val']
    return value

k = funcmin(my_cma.ff.rosen,5*[0],0.8);
print(k['max']['max_val']);