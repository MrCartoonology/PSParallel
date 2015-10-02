import random
from mpi4py import MPI

class MockEvent(object):
    pass

class MockSharedMemServer(object):
    def __init__(self, seed = 23491):
        random.seed(seed*MPI.COMM_WORLD.Get_rank())
        self.remainingEvents = MPI.COMM_WORLD.Get_rank() - 1 # int(round(random.uniform(0,10)))
        print self.remainingEvents

    def events(self):
        while self.remainingEvents > 0:
            self.remainingEvents -= 1
            yield MockEvent()
