import sys
import math, random, json, numpy
import scipy.sparse
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split
from toposort import toposort_flatten
import UncertaintySamplingController
from BaseModel import BaseModel


class RCV1Model(BaseModel):
    def __init__(self, train_size_fine, train_size_coarse, pool_size=None, test_size=None, pond_size=1000):
        # pool size and test size are not used
        main = 'E'
        rcv1_home = 'rcv1'
        all_x, all_z = load_svmlight_file(rcv1_home +'/train.dat')
        test_x, test_z = load_svmlight_file(rcv1_home +'/test.dat')
        self.z_y = json.load(open(rcv1_home +'/full_topics.json'))
        target= main + 'CAT'
        key = target
        cat = set(self.z_y[key])
        topo_data = {}
        hfp = open(rcv1_home + '/topics.hier.dat')
        hfp.readline()
        for line in hfp.readlines():
            pnode, cnode, desc = line.split(' ', 2)
            if not pnode.startswith(main):
                continue
            if pnode not in topo_data:
                topo_data[pnode] = set()
            dest = topo_data[pnode]
            dest.add(cnode)
        self.cat_order = toposort_flatten(topo_data)
        train_x, pool_x, train_z, pool_z = train_test_split(all_x, all_z, train_size=train_size_fine + train_size_coarse)
        train_fine_x, train_coarse_x, train_fine_z, train_coarse_z = train_test_split(train_x, train_z, train_size=train_size_fine)
        pond_x, pool_x, pond_z, pool_z = train_test_split(pool_x, pool_z, train_size=pond_size)

        test_y = np.array([1 if x in cat else 0 for x in test_z])
        train_fine_y = np.array([1 if x in cat else 0 for x in train_fine_z])
        train_coarse_y = np.array([1 if x in cat else 0 for x in train_coarse_z])
        pool_y = np.array([1 if x in cat else 0 for x in pool_z])
        self.training_examples_fine = [train_fine_x, train_fine_z, train_fine_y]
        self.training_examples_coarse = [train_coarse_x, train_coarse_z, train_coarse_y]
        self.pool_examples = [pool_x, pool_z, pool_y]
        self.pond_examples = [pond_x, None, None]
        self.test_examples = [test_x, test_z, test_y]
        self.fine_lr_models = None
        self.coarse_lr_model = None
        print >> sys.stderr, 'Loading done fine=%d coarse=%d pool=%d test=%d pond=%d' % \
                             (self.training_examples_fine[1].shape[0],
                              self.training_examples_coarse[1].shape[0],
                              self.pool_examples[1].shape[0],
                              self.test_examples[1].shape[0],
                              self.pond_examples[0].shape[0])

    def fit(self):
        # train fine models
        self.fine_lr_models = {}
        for key in self.cat_order:
            if key not in self.z_y:
                continue
            cat = set(self.z_y[key])
            train_y = np.array([1 if x in cat else 0 for x in self.training_examples_fine[1]])
            if sum(train_y) < 10 or sum(train_y) == len(train_y):
                continue
            model = LogisticRegression(solver='liblinear', penalty='l2', C=10)
            model.fit(self.training_examples_fine[0], train_y)
            self.fine_lr_models[key] = model

        print >> sys.stderr, 'Fine shape', self.training_examples_fine[0].shape, 'Coarse shape', self.training_examples_coarse[0].shape
        # train coarse model
        model = LogisticRegression(solver='liblinear', penalty='l2', C=10)
        model.fit(scipy.sparse.vstack([self.training_examples_fine[0], self.training_examples_coarse[0]]),
                  list(self.training_examples_fine[2]) + list(self.training_examples_coarse[2]))
        self.coarse_lr_model = model

    def predict_scores(self, examples):
        fine_scores_list = []
        for lr_model_key in self.fine_lr_models:
            fine_scores = self.fine_lr_models[lr_model_key].predict_proba(examples[0])[:, 1]
            fine_scores_list.append(fine_scores)
        coarse_scores = self.coarse_lr_model.predict_proba(examples[0])[:, 1]
        return coarse_scores, fine_scores_list

    def get_pool_size(self):
        return len(self.pool_examples[2])

    def get_test_labels(self):
        return self.test_examples[2]

    def acquire_example_ids(self, ids, example_type):
        if len(ids) > 0:
            ids = set(ids)
            pool_examples = zip(*self.pool_examples)
            selected_pool_examples = [pool_examples[i] for i in range(len(pool_examples)) if i in ids]
            rest_pool_examples = [pool_examples[i] for i in range(len(pool_examples)) if i not in ids]
            self.pool_examples = zip(*rest_pool_examples)
            self.pool_examples[0] = scipy.sparse.vstack(self.pool_examples[0])
            if example_type == 'fine':
                examples = zip(*self.training_examples_fine) + selected_pool_examples
                self.training_examples_fine = [np.array(i) for i in zip(*examples)]
                self.training_examples_fine[0] = scipy.sparse.vstack(self.training_examples_fine[0])
            else:
                examples = zip(*self.training_examples_coarse) + selected_pool_examples
                self.training_examples_coarse = [np.array(i) for i in zip(*examples)]
                self.training_examples_coarse[0] = scipy.sparse.vstack(self.training_examples_coarse[0])
        return


    @staticmethod
    def unittest():
        print 'TESTING [ %s ]' % RCV1Model
        pool_size = 1000
        test_size = 1000
        train_fine_size = 1000
        train_coarse_size = 1000
        model = RCV1Model(train_fine_size, train_coarse_size, pool_size, test_size, pond_size=1000)
        model.fit()
        controller = UncertaintySamplingController.UncertaintySamplingController(model)
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (model.training_examples_fine[0].shape[0],
                                                       model.training_examples_coarse[0].shape[0])
        #Test fine
        pcs_acquire = 10
        print 'ACQUIRE [ %d ] fine examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo='active', method='fine')
        print 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'fine')
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (model.training_examples_fine[0].shape[0],
                                                       model.training_examples_coarse[0].shape[0])
        assert len(model.training_examples_fine[1]) == train_fine_size + pcs_acquire
        #Test Coarse
        pcs_acquire = 10
        print 'ACQUIRE [ %d ] coarse examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo='active', method='coarse')
        print 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'coarse')
        assert len(model.training_examples_coarse[1]) == train_coarse_size + pcs_acquire
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (model.training_examples_fine[0].shape[0],
                                                       model.training_examples_coarse[0].shape[0])

        num_iteration = 4
        budget_step = 200
        for cost in range(3, 4):
            for z in range(1):
                controller = set_up(pond_enabled=True)
                result_buffer = []
                for i in range(num_iteration):
                    metric = controller.current_metric()
                    print >> sys.stderr, 'repeat %d [ cost %d bandit ] iteration %d = %.3f' % (0, cost, i, metric)
                    result_buffer.append(metric)
                    controller.learn_by_cost(budget_step, algo='bandit_uncertainty_sampling', cost_ratio=cost)


def set_up(pond_enabled=False):
    pond_size = 4000
    train_fine_size = 1000
    train_coarse_size = 1000
    model = RCV1Model(train_size_fine=train_fine_size, train_size_coarse=train_coarse_size, pond_size=pond_size)
    model.fit()
    controller = UncertaintySamplingController.UncertaintySamplingController(model, pond_enabled=pond_enabled, reward_factor=300.0)
    return controller

def many_types():
    pcs_acquire = 200
    num_iteration = 90
    repeats = 1
    result = []
    for j in range(repeats):
        for method in ['coarse', 'fine']:
            for algo in ['active', 'passive']:
                controller = set_up()
                result_buffer = []
                for i in range(num_iteration):
                    metric = controller.current_metric()
                    print >> sys.stderr, 'repeat %d [ %s  %s ] iteration %d' % (j, method, algo, i)
                    result_buffer.append(metric)
                    pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo=algo, method=method)
                    controller.model.acquire_example_ids(pcs, method)
                    controller.model.fit()
                result.append(result_buffer)
        print json.dumps(result)
    return

def dynamic_ratio():
    num_iteration = 90
    budget_step = 200
    repeats = 1
    result = []
    for j in range(repeats):
        for cost in range(1,10):
            for z in range(1):
                controller = set_up(pond_enabled=True)
                result_buffer = []
                for i in range(num_iteration):
                    metric = controller.current_metric()
                    print >> sys.stderr, 'repeat %d [ cost %d bandit ] iteration %d = %.3f' % (j, cost, i, metric)
                    result_buffer.append(metric)
                    controller.learn_by_cost(budget_step, algo='bandit_uncertainty_sampling', cost_ratio=cost)
                result.append(result_buffer)
            # continue
            for ratio in range(11):
                ratio = 0.1 * ratio
                controller = set_up()
                result_buffer = []
                for i in range(num_iteration):
                    metric = controller.current_metric()
                    print >> sys.stderr, 'repeat %d [ cost %d ratio %.2f ] iteration %d = %.3f' % (j, cost, ratio, i, metric)
                    result_buffer.append(metric)
                    controller.learn_by_cost(budget_step, algo='fixed_fine_ratio', cost_ratio=cost, fixed_fine_budget_ratio=ratio )
                result.append(result_buffer)
        print json.dumps(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t')
    args = parser.parse_args()
    if args.t == 'unittest':
        RCV1Model.unittest()
    if args.t == 'many_types':
        many_types()
    if args.t == 'dynamic_ratio':
        dynamic_ratio()