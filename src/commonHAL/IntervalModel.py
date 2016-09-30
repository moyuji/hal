import math, random, numpy, sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import UncertaintySamplingController
import json
import argparse
from BaseModel import BaseModel

class IntervalModel(BaseModel):
    num_layers = 3
    def __init__(self, train_size_fine, train_size_coarse, pool_size, test_size, pond_size=1000, err=0.1):
        self.training_examples_fine = self.generate_random_interval_examples(train_size_fine, err)
        self.training_examples_coarse = self.generate_random_interval_examples(train_size_coarse, err)
        self.pool_examples = self.generate_random_interval_examples(pool_size, err)
        self.pond_examples = self.generate_random_interval_examples(pond_size, err)
        self.test_examples = self.generate_random_interval_examples(test_size, err)
        self.fine_gbr_models = None
        self.coarse_gbr_model = None
        return

    def fit(self):
        # train fine models
        self.fine_gbr_models = []
        for i in range(1, 18, 2):
            y = []
            for (j, v) in enumerate(self.training_examples_fine[1]):
                if v != i and self.training_examples_fine[2][j] == 0:
                    y.append(0)
                else:
                    if v == i:
                        y.append(1)
                    else:
                        y.append(0)
            gbr_model = GradientBoostingRegressor(n_estimators=30,
                                                  learning_rate=0.9,
                                                  max_depth=1,
                                                  subsample=0.8)
            gbr_model.fit(self.training_examples_fine[0], y)
            self.fine_gbr_models.append(gbr_model)

        # train coarse model
        self.coarse_gbr_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.9, max_depth=1, subsample=0.8)
        self.coarse_gbr_model.fit(self.training_examples_fine[0] + self.training_examples_coarse[0],
                                  self.training_examples_fine[2] + self.training_examples_coarse[2])

    def acquire_example_ids(self, ids, example_type):
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
        return

    @staticmethod
    def normalize_scores(scores):
        return [min(1.0, max(v, 0.0)) for v in scores]

    def predict_scores(self, examples):
        fine_scores_list = []
        for grb_model in self.fine_gbr_models:
            fine_scores = self.normalize_scores(grb_model.predict(examples[0]))
            fine_scores_list.append(fine_scores)
        coarse_scores = self.normalize_scores(self.coarse_gbr_model.predict(examples[0]))
        return coarse_scores, fine_scores_list

    def get_pool_size(self):
        return len(self.pool_examples[0])

    def get_test_labels(self):
        return self.test_examples[2]

    """
    :return [features, fine_categories, coarse_labels]
    """
    @staticmethod
    def generate_random_interval_examples(num, p):
        n = 2
        k = IntervalModel.num_layers
        eg = [[], [], []]
        for i in range(0, num):
            v = random.uniform(0,k**n*2)
            c = math.floor(v)
            if random.uniform(0,1) < p:
                while True:
                    s = random.uniform(0,k**n*2)
                    if math.floor(s)%2 == c%2:
                        continue
                    c = math.floor(s)
                    break
            if c%2 == 0 :
                c = 0
                eg[2].append(0)
            else:
                eg[2].append(1)
            eg[1].append(c)
            eg[0].append([v])
        # eg[0] random(0, 18) eg[1] fine labels eg[2] coarse labels
        return eg

    @staticmethod
    def unittest():
        print 'TESTING [ %s ]' % IntervalModel
        pool_size = 1230
        test_size = 100
        train_fine_size = 20
        train_coarse_size = 20
        model = IntervalModel(train_fine_size, train_coarse_size, pool_size, test_size)
        model.fit()
        coarse_scores, fine_scores = model.score('pool')
        assert len(fine_scores) == 9, 'wrong number of fine classifiers'
        assert len(coarse_scores) == pool_size, 'mismatched pool_size'

        controller = UncertaintySamplingController.UncertaintySamplingController(model)
        assert max(coarse_scores) <= 1.0
        assert min(coarse_scores) >= 0.0
        print 'Metric = %f' % controller.current_metric()
        print 'ALL Unc AVG = %f' % np.mean(controller.current_uncertainty(data_group='test'))

        #Begin active learning test
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (len(model.training_examples_fine[0]),
                                                       len(model.training_examples_coarse[0]))
        #Test fine
        pcs_acquire = 10
        print 'ACQUIRE [ %d ] fine examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo='active', method='fine')
        print 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'fine')
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (len(model.training_examples_fine[0]),
                                                       len(model.training_examples_coarse[0]))
        assert len(model.training_examples_fine[0]) == train_fine_size + pcs_acquire
        #Test Coarse
        pcs_acquire = 10
        print 'ACQUIRE [ %d ] coarse examples' % pcs_acquire
        pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo='active', method='coarse')
        print 'Recommendation Uncertainty AVG = %f' % np.mean(ucties)
        assert np.mean(ucties) > 0
        model.acquire_example_ids(pcs, 'coarse')
        assert len(model.training_examples_coarse[0]) == train_coarse_size + pcs_acquire
        print 'Fine Size [ %d ] Coarse Size [ %d ]' % (len(model.training_examples_fine[0]),
                                                       len(model.training_examples_coarse[0]))
        many_types(num_iteration=2)
        dynamic_ratio(num_iteration=2)

        print 'PASSED [ %s ]' % IntervalModel
        return controller


def set_up(pond_enabled=False):
    pool_size = 10000
    test_size = 8000
    pond_size = 20000
    train_fine_size = 50
    train_coarse_size = 50
    model = IntervalModel(train_size_fine=train_fine_size,
                          train_size_coarse=train_coarse_size,
                          pool_size=pool_size,
                          test_size=test_size,
                          pond_size=pond_size)
    model.fit()
    controller = UncertaintySamplingController.UncertaintySamplingController(model,
                                                                             pond_enabled=pond_enabled,
                                                                             reward_factor=400.0)
    return controller


def many_types(num_iteration=300, repeats=5):
    pcs_acquire = 4
    result = []
    for j in range(repeats):
        for method in ['coarse', 'fine']:
            for algo in ['active', 'passive']:
                controller = set_up()
                result_buffer = []
                for i in range(num_iteration):
                    metric = controller.current_metric()
                    print >> sys.stderr, 'repeat %d [ %s %s ] iteration %d' % (j, method, algo, i)
                    result_buffer.append(metric)
                    pcs, ucties = controller.recommend_acquisition_ids(pcs_acquire, algo=algo, method=method)
                    controller.model.acquire_example_ids(pcs, method)
                    controller.model.fit()
                result.append(result_buffer)
        print json.dumps(result)


def dynamic_ratio(num_iteration=300, repeats=1, budget_step=4):
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
        IntervalModel.unittest()
    if args.t == 'many_types':
        many_types()
    if args.t == 'dynamic_ratio':
        dynamic_ratio()