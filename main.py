import my_cma

es = my_cma.CMAEvolutionStrategy(0.45019955799280803, 0.047292304159400896, 0.04785904960298248, 5 * [0], 0.8, {'seed':400000})
self = es.optimize(my_cma.ff.rosen)
print(self.result[1])