'''
With shared memory, the number of events each worker processes can differ
and is unknown ahead of time. This makes collective operations like reduce
more involved than in the case where each rank will handle exactly the same
number of collective operations. A technique to manage this is to update 
the communicatior used as different ranks drop out earlier than other ranks. 

This example implements a simpler viewer/worker MPI program where a historgram
is periodically reduced. The workers and viewers use the packged function 
collectiveCommUpdate before each reduce, as well as after all worker events 
are processed, in order to update the communicator.
'''

import numpy as np
from mpi4py import MPI

from PSParallel.CommUpdate import collectiveCommUpdate
from PSParallel.Util import dprint

class MockEvent(object):
    pass

class MockDataSource(object):
    '''provides mock event iterator over F * (R-1) events, where
    R is the MPI workd rank, and F is a parameterizable factor. 
    
    Rank 0 or 1 will provide 0 events.
    '''
    def __init__(self, numEventsScaleFactor=1):
        self.remainingEvents = max(0,numEventsScaleFactor*(MPI.COMM_WORLD.Get_rank()-1))
        dprint("MockDataSource with %d events" % self.remainingEvents)

    def events(self):
        while self.remainingEvents > 0:
            self.remainingEvents -= 1
            yield MockEvent()

def runViewerReduceHistorgram(comm, histogramSize, finalReducedHistFilename):
    '''Viewer shared memory example for reducing a histogram from worker ranks.
   
    Maintains a histogram. Continues to update the communicator and reduce until
    no more workers participate in the communicator.

    Assumes Each reduce is a histogram over only the previous worker events since 
    the last reduce.

    ARGS:
      comm - initial communicator to use
      histogramSize - size of historgram to reduce
      finalReducedHistFilename - filename for numpy .npy file for final histogram
    '''
    assert comm.Get_rank()==0
    histogram = np.zeros(histogramSize, np.int64)
    while True:
        comm = collectiveCommUpdate(comm, stayInComm=True)

        # check if only the viewer is left in communicator and exit if so
        if comm.Get_size() == 1: break

        localHistogram = np.zeros(histogramSize, np.int64)
        recvHistogram = np.zeros(histogramSize, np.int64)
        dprint("before Reduce")
        comm.Reduce(sendbuf=[localHistogram, MPI.INT64_T], 
                    recvbuf=[recvHistogram, MPI.INT64_T], 
                    op=MPI.SUM, root=0)
        histogram[:] += recvHistogram[:]
        print "rank=0 After Reduce. reduced histogram: %s" % histogram
    np.save(file(finalReducedHistFilename,'w'), histogram)
    print "rank=0 saved final reduced historgram=%s in file: %s" % (histogram, finalReducedHistFilename)

def workerReduce(comm, localHistogram):
    dprint("before Reduce sending=%s" % localHistogram)
    comm.Reduce(sendbuf=[localHistogram, MPI.INT64_T], 
                recvbuf=[None, MPI.INT64_T], 
                op=MPI.SUM, root=0)
    dprint("after reduce")

def runWorker(comm, histogramSize, gatherInterval, numEventsScaleFactor):
    '''worker shared memory example for reducing a histogram to a viewer
   
    Maintains a histogram. Continues to update the communicator and reduce until
    no more workers participate in the communicator.

    Assumes Each reduce is a histogram over only the previous worker events since 
    the last reduce.

    ARGS:
      comm - initial communicator to use
      histogramSize - size of historgram to reduce
      finalReducedHistFilename - filename for numpy .npy file for final histogram
    '''
    sharedMemDs = MockDataSource(numEventsScaleFactor)
    localHistogram = np.zeros(histogramSize, np.int64)
    idx = None
    for idx, evt in enumerate(sharedMemDs.events()):
        localHistogram[idx % histogramSize] += 1
        if (idx % gatherInterval) == 0:
            comm = collectiveCommUpdate(comm, stayInComm=True)
            workerReduce(comm, localHistogram)
            localHistogram[:] = 0 

    # check for last reduce
    if (idx is not None) and (idx % gatherInterval) != 0:
        comm = collectiveCommUpdate(comm, stayInComm=True)
        workerReduce(comm, localHistogram)

    # remove self from collective communication
    dprint("before final comm update to remove self")
    collectiveCommUpdate(comm, stayInComm=False)
    dprint("after final comm update")

def sharedMemoryExample(comm, gatherInterval, 
                        histogramSize, finalReducedHistFilename, 
                        numEventsScaleFactor):

    if comm.Get_rank()==0:
        runViewerReduceHistorgram(comm, histogramSize, finalReducedHistFilename)
    else:
        runWorker(comm, histogramSize, gatherInterval, numEventsScaleFactor)


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    assert comm.Get_size()>=2, "need at least two ranks for example"
    histogramSize = 5
    finalReducedHistFilename = 'finalReduedHist.npy'
    numEventsScaleFactor = 3
    sharedMemoryExample(comm, 
                        gatherInterval=4, 
                        histogramSize=histogramSize,
                        finalReducedHistFilename=finalReducedHistFilename,
                        numEventsScaleFactor=numEventsScaleFactor)

    # do a test to make sure we get the expected answer:
    if comm.Get_rank()==0:
        reducedHist = np.load(file(finalReducedHistFilename,'r'))
        expectedAnswer = np.zeros(histogramSize, np.int64)
        for rank in range(1,comm.Get_size()):
            numEvents = max(0,numEventsScaleFactor * (rank-1))
            for idx in range(numEvents):
                expectedAnswer[idx % histogramSize] += 1
        if not np.all(expectedAnswer == reducedHist):
            print "ERROR: reduced histogram != expected answer: %s" % expectedAnswer
        else:
            print "SUCCESS: reduced histogram agrees with expected anser: %s" % expectedAnswer

