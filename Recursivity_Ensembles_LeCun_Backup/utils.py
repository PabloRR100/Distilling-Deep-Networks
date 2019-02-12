
import time

# GENERAL USE
# -----------

class Results():
    ''' Object to store training / validation results'''
    def __init__(self):
        super(Results, self).__init__()
        self.train_loss = [] 
        self.train_accy = []
        self.valid_loss = []
        self.valid_accy = []


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


# Count parameters of a model 
def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())