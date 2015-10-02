# PSParallel
This is a package for the psana framework used to analyze LCLS data. 
Documentation about data analysis at LCLS  starts here: 

https://confluence.slac.stanford.edu/display/PSDM/LCLS+Data+Analysis

and more details about the software environment, psana framework and how
packages fit into it and are built is here:

https://confluence.slac.stanford.edu/display/PSDM/Analysis+Environment+Details
https://confluence.slac.stanford.edu/display/PSDM/Packages+and+Releases
https://confluence.slac.stanford.edu/display/PSDM/SConsTools

The purpose of this package is to solve a problem users have when using MPI to
analyze events from shared memory. When reading from shared memory, each MPI rank
can get a different number of events. This makes intermittent collective communication
such as gather or reduce more complicated. The complication is that one rank
may only get enough data to participate in say 5 of these gathers, while another will
particpate in 6 of these gathers. The solution presented here is to always update
the communicator used for collective communication before a gather or reduce, allowing
ranks which process fewer events to drop out of the collective communication.
