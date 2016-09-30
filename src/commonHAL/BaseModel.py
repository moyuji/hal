import sys
import resource
import time
class BaseModel:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()
        return
    @staticmethod
    def use_custom_agg():
       return False
    def get_data_group(self, data_group):
        if data_group == 'test':
            ex = self.test_examples
        elif data_group == 'pond':
            ex = self.pond_examples
        elif data_group == 'pool':
            ex = self.pool_examples
        else:
            assert False, 'Unknown data_group  [ %s ]' % data_group
        return ex
    def score(self, data_group):
        return self.predict_scores(self.get_data_group(data_group))
    def track_performance(self, flag=''):
        print >> sys.stderr, '%s Memory Usage [ %.3f GB ] CPU Usage [ %.5f Hr / %.5f Hr ] ' % \
                             (flag,
                              1.0 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,
                              (time.time() - self.last_time)/3600,
                              (time.time() - self.start_time)/3600)
        self.last_time = time.time()



