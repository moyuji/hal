import sys
import os
import cPickle as pickle
import pycrfsuite
import random
from collections import Counter
import argparse
import numpy as np
import UncertaintySamplingController
from sklearn.cross_validation import train_test_split
from BaseModel import BaseModel
from itertools import chain
import concurrent.futures

workbase = '/tmp/crfmodel/%d' %random.randint(0,100000000)
os.makedirs(workbase)
def getTrainer():
    crfpar={
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 40,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    }
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params(crfpar)
    return trainer

def traincrf(feats, labels, tag):
    crfpar={
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 40,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    }
    crf = pycrfsuite.Trainer(verbose=False)
    crf.set_params(crfpar)
    for xseq, yseq in zip(feats, labels):
        crf.append(xseq, yseq)
    crf.train('%s/%s.crf' % (CRFModelHelper.workbase, tag))
    return 0

def predictcrf(feats, tag):
    y_score = []
    tagger = pycrfsuite.Tagger()
    tagger.open('%s/%s.crf' % (CRFModelHelper.workbase, tag))
    for xseq in feats:
        lseq = tagger.tag(xseq)
        mags = []
        for i in range(len(lseq)):
            mag = tagger.marginal(lseq[i], i)
            if lseq[i] == 'token':
                mag = 1.0 - mag
            mags.append(mag)
        y_score += mags
    return y_score

class CRFModelHelper():
    workbase = '/tmp/crfmodel/%d' %random.randint(0,100000000)
    os.makedirs(workbase)
    @staticmethod
    def getTrainer():
        crfpar={
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 40,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        }
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params(crfpar)
        return trainer

    @staticmethod
    def traincrf(feats, labels, tag):
        crf=CRFModelHelper.getTrainer()
        for xseq, yseq in zip(feats, labels):
            crf.append(xseq, yseq)
        crf.train('%s/%s.crf' % (CRFModelHelper.workbase, tag))

    @staticmethod
    def predictcrf(feats, tag):
        y_score = []
        tagger = pycrfsuite.Tagger()
        tagger.open('%s/%s.crf' % (CRFModelHelper.workbase, tag))
        for xseq in feats:
            lseq = tagger.tag(xseq)
            mags = []
            for i in range(len(lseq)):
                mag = tagger.marginal(lseq[i], i)
                if lseq[i] == 'token':
                    mag = 1.0 - mag
                mags.append(mag)
            y_score += mags
        return y_score


class RichmondModel(BaseModel):
    n_thread = 1
    @staticmethod
    def use_custom_agg():
        return True

    def __init__(self, train_size_fine, train_size_coarse, pool_size=None, test_size=None, pond_size=1000):
        BaseModel.__init__(self)
        print >> sys.stderr, 'Loading features and Labels'
        feats, labels, fines = pickle.load(open('richmond_data/ex.pickle'))
        train_x, pool_x, train_z, pool_z, train_y, pool_y = \
            train_test_split(feats, fines, labels, train_size=train_size_fine + train_size_coarse)
        train_fine_x, train_coarse_x, train_fine_z, train_coarse_z, train_fine_y, train_coarse_y = \
            train_test_split(train_x, train_z, train_y, train_size=train_size_fine)
        pond_x, pool_x, pond_z, pool_z, pond_y, pool_y= train_test_split(pool_x, pool_z, pool_y, train_size=pond_size)
        test_x, pool_x, test_z, pool_z, test_y, pool_y= train_test_split(pool_x, pool_z, pool_y, train_size=test_size)
        if pool_size is not None:
            pool_x, _, pool_z, _, pool_y, _ = train_test_split(pool_x, pool_z, pool_y, train_size=pool_size)

        self.training_examples_fine = [train_fine_x, train_fine_z, train_fine_y]
        self.training_examples_coarse = [train_coarse_x, train_coarse_z, train_coarse_y]
        self.pool_examples = [pool_x, pool_z, pool_y]
        self.pond_examples = [pond_x, None, None]
        self.test_examples = [test_x, test_z, test_y]
        self.fine_crf_models = None
        self.coarse_crf_model = None
        print >> sys.stderr, 'Loading done fine=%d coarse=%d pool=%d test=%d pond=%d total=%d' % \
                             (len(self.training_examples_fine[1]),
                              len(self.training_examples_coarse[1]),
                              len(self.pool_examples[1]),
                              len(self.test_examples[1]),
                              len(self.pond_examples[0]),
                              len(labels))
    @staticmethod
    def masklabel(labels, mask = None):
        new_labels = []
        for label in labels:
            if mask is None:
                new_labels.append(['orgname' if v != 'token' else v for v in label])
            else:
                new_labels.append([ mask if v == mask else 'token' for v in label])
        return new_labels

    def fit(self):
        self.track_performance('training started')

        self.fine_crf_models = []
        fine_tags_counter = Counter(chain(*self.training_examples_fine[1]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_thread) as e:
            self.coarse_crf_model = 'coarse'
            e.submit(traincrf, self.training_examples_fine[0] + self.training_examples_coarse[0],
                      self.training_examples_fine[2] + self.training_examples_coarse[2], self.coarse_crf_model)
            for fine_tag in fine_tags_counter:
                if fine_tag == 'token' or fine_tags_counter[fine_tag] <=5:
                    continue
                fine_crf_model = 'fine_' + fine_tag
                e.submit(traincrf, self.training_examples_fine[0],
                         self.masklabel(self.training_examples_fine[1], mask=fine_tag), fine_crf_model)
                # CRFModelHelper.traincrf(self.training_examples_fine[0],
                #                         self.masklabel(self.training_examples_fine[1], mask=fine_tag), fine_crf_model)
                self.fine_crf_models.append(fine_crf_model)
            e.shutdown()
        self.track_performance('training finished')

    def predict_scores(self, examples):
        self.track_performance('evaluation started')
        fine_scores_list = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_thread) as e:
            futures.append(e.submit(predictcrf, examples[0], self.coarse_crf_model))
            for crf_model_key in self.fine_crf_models:
                futures.append(e.submit(predictcrf, examples[0], crf_model_key))
                #fine_scores = CRFModelHelper.predictcrf(examples[0], crf_model_key)
                #fine_scores_list.append(fine_scores)
            coarse_scores = futures[0].result()
            futures.pop(0)
            for fine_future in futures:
                fine_scores_list.append(fine_future.result())
            #coarse_scores = CRFModelHelper.predictcrf(examples[0], self.coarse_crf_model)
        self.track_performance('evaluation finished')
        return coarse_scores, fine_scores_list

    def get_pool_size(self):
        return len(self.pool_examples[2])

    def predict_pool_scores(self):
        return self.predict_scores(self.pool_examples)

    def predict_test_scores(self):
        return self.predict_scores(self.test_examples)

    def get_test_labels(self):
        result = []
        for sentence in self.test_examples[2]:
            result += [0.0 if v == 'token' else 1.0 for v in sentence ]
        return result

    def predict_pond_scores(self):
        return self.predict_scores(self.pond_examples)

    def aggregate_uncertainty(self, ucties, data_group):
        result = []
        if data_group == 'test':
            examples = self.test_examples[0]
        elif data_group == 'pond':
            examples = self.pond_examples[0]
        elif data_group == 'pool':
            examples = self.pool_examples[0]
        else:
            assert False, 'Unknown data_group  [ %s ]' % data_group
        b = 0
        for ex in examples:
            e = b + len(ex)
            result.append(sum(ucties[b:e]))
            b = e
        return result

    def acquire_example_ids(self, ids, example_type):
        self.track_performance('Acquiring started')
        if len(ids) > 0:
            ids = set(ids)
            pool_examples = zip(*self.pool_examples)
            selected_pool_examples = [pool_examples[i] for i in range(len(pool_examples)) if i in ids]
            rest_pool_examples = [pool_examples[i] for i in range(len(pool_examples)) if i not in ids]
            self.pool_examples = zip(*rest_pool_examples)
            if example_type == 'fine':
                examples = zip(*self.training_examples_fine) + selected_pool_examples
                self.training_examples_fine = [list(i) for i in zip(*examples)]
            else:
                examples = zip(*self.training_examples_coarse) + selected_pool_examples
                self.training_examples_coarse = [list(i) for i in zip(*examples)]
        print >> sys.stderr, 'Fine size [ %d ] Coarse size [ %d ]' % \
                             (len(self.training_examples_fine[0]), len(self.training_examples_coarse[0]))
        self.track_performance('Acquiring finished')
        return

    @staticmethod
    def unittest():
        print 'Start testing [ %s ]' % RichmondModel
        pool_size = 900
        test_size = 700
        train_fine_size = 500
        train_coarse_size = 600
        model = RichmondModel(train_fine_size, train_coarse_size, pool_size, test_size, pond_size=1000)
        model.fit()
        controller = UncertaintySamplingController.UncertaintySamplingController(model)
        print >> sys.stderr, 'Metric = %f' % controller.current_metric()
        print >> sys.stderr, 'ALL Unc AVG = %f' % np.mean(controller.current_uncertainty(data_group='test'))

        #Begin active learning test
        print >> sys.stderr, 'Fine Size [ %d ] Coarse Size [ %d ]' % (len(model.training_examples_fine[0]),
                                                       len(model.training_examples_coarse[0]))
        #Test fine
        pcs_acquire = 10
        print >> sys.stderr, 'ACQUIRE [ %d ] fine examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(
            pcs_acquire, algo='active', method='fine')
        print >> sys.stderr, 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'fine')
        print >> sys.stderr, 'Fine Size [ %d ] Coarse Size [ %d ]' % (len(model.training_examples_fine[0]),
                                                       len(model.training_examples_coarse[0]))
        assert len(model.training_examples_fine[0]) == train_fine_size + pcs_acquire
        #Test Coarse
        pcs_acquire = 10
        print >> sys.stderr, 'ACQUIRE [ %d ] coarse examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(
            pcs_acquire, algo='active', method='coarse')
        print >> sys.stderr, 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'coarse')
        assert len(model.training_examples_coarse[0]) == train_coarse_size + pcs_acquire
        print >> sys.stderr, 'Fine Size [ %d ] Coarse Size [ %d ]' % \
                             (len(model.training_examples_fine[0]), len(model.training_examples_coarse[0]))
        controller.learn_by_cost(budget_size=25, algo='fixed_fine_ratio',
                                 cost_ratio=1.0, fixed_fine_budget_ratio=0.0)
        print >> sys.stderr, 'Fine Size [ %d ] Coarse Size [ %d ]' % \
                             (len(model.training_examples_fine[0]), len(model.training_examples_coarse[0]))

        controller = set_up(pond_enabled=True)
        for i in range(5):
            controller.current_metric()
            controller.learn_by_cost(100, algo='bandit_uncertainty_sampling', cost_ratio=1.0)
        print >> sys.stderr, 'EOF testing [ %s ]' % RichmondModel

def set_up(pond_enabled=False):
    pool_size = 10000
    test_size = 20000
    pond_size = 10000
    train_fine_size = 800
    train_coarse_size = 800
    model = RichmondModel(train_size_fine=train_fine_size,
                          train_size_coarse=train_coarse_size,
                          pool_size=pool_size,
                          test_size=test_size,
                          pond_size=pond_size)
    model.fit()
    controller = UncertaintySamplingController.UncertaintySamplingController(model,
                                                                             pond_enabled=pond_enabled,
                                                                             reward_factor=2000.0)
    return controller

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t')
    args = parser.parse_args()
    if args.t == 'unittest':
        RichmondModel.unittest()
