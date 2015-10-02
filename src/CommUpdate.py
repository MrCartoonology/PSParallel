from PSParallel.Util import dprint

def collectiveCommUpdate(comm, stayInComm):    
    '''collectively update a communicator based on which ranks continue.

    Note: requires rank 0 to be the root/viewer/master. 
    Rank 0 must always pass stayInComm=True

    ARGS:
      comm - MPI communicator
      stayInComm - Bool - true if this rank will stay in the communicator

    RETURN:
      comm - a potentially new communicator based on ranks that stayed in,
             however will be existing comm if no change.
             comm will always have at least one rank in it (rank 0)

    Example:
      comm = collectiveCommUpdate(comm, True)
    '''
    dprint("collectiveCommUpdate start: before gathering stayInComm at root. Sending %r" % stayInComm)
    if comm.Get_rank()==0:
        assert stayInComm == True  # root always stays in comm
        recvObject = comm.gather(sendobj=True, root=0)
    else:
        comm.gather(sendobj=stayInComm, root=0)

    dprint("  after gather. Before bcast to inform all in old communicatior of ranks to drop")
    if comm.Get_rank()==0:
        droppedRanks = [idx for idx, hasData in enumerate(recvObject) if not hasData]
        comm.bcast(obj=droppedRanks, root=0)
    else:
        droppedRanks = comm.bcast()

    dprint("  after bcast. droppedRanks=%r" % droppedRanks)
    if len(droppedRanks)==0:
        return comm

    assert 0 not in droppedRanks
    group = comm.Get_group()
    newGroup = group.Excl(droppedRanks)
    dprint("  before comm.Create(newGroup) where newGroup excludes ranks %s" % droppedRanks)
    newComm = comm.Create(newGroup)
    dprint("  after comm.Create(newGroup)")
    return newComm
