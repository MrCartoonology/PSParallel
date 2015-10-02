DEBUG = False

def dprint(msg, debug=DEBUG):
    if debug:
        print "rank=%d: %s" % (MPI.COMM_WORLD.Get_rank(), msg)
