from random import uniform, randint
from copy import deepcopy

class ReservoirSampler:
    
    def __init__(self, max_samples):
        self.max_samples = max_samples
        self.num_samples = 0
        self.reservoir = dict.fromkeys(range(1, self.max_samples+1))
    
    def NewSamples(self, samples):
        for sample in samples:
            self.num_samples+=1
            if self.num_samples <= self.max_samples:
                self.reservoir[self.num_samples] = deepcopy(sample)
            else:
                prob_add = float(self.max_samples) / float(self.num_samples)
                if uniform(0,1) <= prob_add:
                    self.reservoir[randint(1,self.max_samples)] = deepcopy(sample)
    
    def ReturnReservoir(self):
        reservoir = []
        for sample_id in self.reservoir:
            reservoir.append(self.reservoir[sample_id])
        return reservoir