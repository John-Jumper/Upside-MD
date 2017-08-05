from mpi4py import MPI
import threading

# A pretty serious flaw is the assumption that code is loaded in the same order on all ranks
# I could fix this by going to names instead

# FIXME register a method to remove objects when __del__ is called on rank 0
# Be careful that the MpiCollectiveObject class only holds a weakref on rank 0
# FIXME carefully enable pickling support

class MpiCollectiveObject(object):
    '''A special class for creating objects on all ranks, so that decorated methods are called collectively'''
    def __init__(self, comm, wait_granularity=0.2):
        self.comm = comm

        # Wait granularity is the amount of time to wait between waking to check for new 
        # messages.  This prevents the default busy-waiting behavior of MPI.
        self.wait_granularity = float(wait_granularity)

        if not self.comm.rank:
            # Only the master needs protection with a lock
            # Locks are necessary because tensorflow using threading by default when evaluating 
            # the computation graph
            self.lock = threading.Lock()
        self.objects = []
        self.methods = []
        self.classes = []

    def start_worker_loop(self):
        import time

        # Rank 0 does not participate
        if not self.comm.rank: return

        while True:
            # First we will use a non-blocking barrier to avoid busy-waiting for messages
            barrier_request = self.comm.Ibarrier()
            while not barrier_request.Get_status():
                time.sleep(self.wait_granularity)

            # post bcast for all nodes to receive message from rank 0
            object_index, method_index, args, kwargs = self.comm.bcast(None)
            if object_index is None:  # special value to call constructor
                class_index, object_index = method_index
                assert len(self.objects) == object_index
                self.objects.append(self.classes[class_index](*args, **kwargs))
            else:
                self.methods[method_index](self.objects[object_index], *args, **kwargs)

    # This function is intended to be used as a method decorator
    def collective(self, f):
        # Record a number for the method so it is easy to call collectively
        method_index = len(self.methods)
        self.methods.append(f)

        # On any rank other than 0, the function should be unadulterated because it is 
        # already being called within a collective remote procedure call
        if self.comm.rank:
            return f
        else:
            def wrap(self_from_call, *args, **kwargs):
                # we need to discard the self_from_call argument during the RPC because this is a self
                # on rank 0 not the other ranks.  We use the object number as its replacement
                with self.lock:
                    # First send the wake-up from barrier message
                    self.comm.Ibarrier().Wait()

                    # Now send the RPC message to start the workers
                    self.comm.bcast((self_from_call.mpi_object_index, method_index, args, kwargs))
                    return f(self_from_call, *args, **kwargs)
            return wrap

    def register(self, c):
        c.class_index = len(self.classes)
        self.classes.append(c)
        old_init = c.__init__

        def wrap_init(self_from_call, *args, **kwargs):
            self_from_call.comm = self.comm

            if not self.comm.rank:
                import weakref
                with self.lock:
                    # We only append a weakref proxy from rank 0 so that this class does not keep
                    # objects alive on rank 0.  FIXME implement removing the object on all other threads
                    # using weakref callback.
                    self_from_call.mpi_object_index = len(self.objects)
                    self.objects.append(weakref.proxy(self_from_call))

                    self.comm.Ibarrier().Wait()
                    self.comm.bcast((
                        None,
                        (c.class_index,self_from_call.mpi_object_index),
                        args,
                        kwargs))
                    
            old_init(self_from_call, *args, **kwargs)

        c.__init__ = wrap_init
        return c



# We only want a singleton of the collective object
# The user should not call begin on the singleton until all code is declared
obj = MpiCollectiveObject(MPI.COMM_WORLD)

def test_main():
    import numpy as np

    @holder.register
    class TestCollective(object):
        def __init__(self, test_array):
            self.test_array = test_array
    
        @obj.collective
        def identify_yourself(self):
            print 'I am rank %i holding %s'%(self.comm.rank, self.test_array)
            self.comm.barrier()
    
        @obj.collective
        def reduce(self):
            return self.comm.reduce(self.test_array)
    
        def non_mpi_function(self):
            print 'I am rank %i'%self.comm.rank
    
        @obj.collective
        def increment_by_rank(self, factor):
            self.test_array += factor*self.comm.rank

    obj.start_worker_loop()

    print 'hello everyone message'
    obj1 = TestCollective(np.zeros((3,)))
    obj2 = TestCollective(np.ones((3,)))
    print
    print 'identify_yourself'
    obj1.identify_yourself()
    obj2.identify_yourself()
    print
    print 'increment_by_rank 0.1'
    obj1.increment_by_rank(0.1)
    obj2.increment_by_rank(0.1)
    print
    print 'identify_yourself'
    obj1.identify_yourself()
    obj2.identify_yourself()
    print
    print 'Non-mpi function'
    obj1.non_mpi_function()
    obj2.non_mpi_function()
    print
    print 'Total array'
    print obj1.reduce()
    print obj2.reduce()
    print
    sys.stdout.flush()
    MPI.COMM_WORLD.Abort()
    
if __name__ == '__main__':
    test_main()


