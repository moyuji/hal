import IntervalModel
import sklearn.metrics
import numpy as np
import random
import math
import sys
import resource
import time
import scipy as sp
from scipy import stats
from statsmodels.stats.proportion import proportion_confint as confint
class UncertaintySamplingController:
    def __init__(self, baseModel, pond_enabled=False, reward_factor=1.0):
        self.model = baseModel
        self.reward_factor = reward_factor
        self.start_time = time.time()
        self.last_time = time.time()
        if pond_enabled:
            self.pond_enabled = pond_enabled
            self.pond_scores = self.score_alpha_beta(self.model.score(data_group='pond'))
            self.fine_rewards = []
            self.coarse_rewards = []
            self.coarse_coarse_rewards = []
            self.fine_fine_rewards = []
            self.fine_coarse_rewards = []
            self.coarse_fine_rewards = []
            self.last_played = None
            self.last_reward = None

    def track_performance(self):
        print >> sys.stderr, 'Memory Usage [ %.3f GB ] CPU Usage [ %.5f Hr / %.5f Hr ] ' % \
                             (1.0 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024,
                              (time.time() - self.last_time)/3600,
                              (time.time() - self.start_time)/3600)
        self.last_time = time.time()

    def logloss(self, act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        return ll

    def compute_reward(self):
        assert self.pond_enabled, 'pond not enabled'
        new_scores = self.score_alpha_beta(self.model.score(data_group='pond'))
        reward = np.mean([math.log(abs(i-j)) for (i, j) in zip(self.pond_scores, new_scores)])
        #reward = stats.pearsonr(new_scores, self.pond_scores)[0]
        self.pond_scores = new_scores
        print >> sys.stderr, 'rewards = %.4f' % reward
        return reward

    def current_metric(self):
        self.track_performance()

        return self.current_pr_auc()

    def current_pr_auc(self):
        labels = self.model.get_test_labels()
        preds = self.score_alpha_beta(self.model.score('test'))
        auc = sklearn.metrics.average_precision_score(labels, preds)
        return auc

    def uncertainty_alpha_beta(self, method, data_group):
        coarse_scores, fine_scores_list = self.model.score(data_group)
        if method == 'hybrid':
            preds = self.score_alpha_beta((coarse_scores, fine_scores_list))
        elif method == 'coarse':
            preds = self.score_alpha_beta((coarse_scores, []))
        elif method == 'fine':
            preds = self.score_alpha_beta(([], fine_scores_list))
        else:
            assert False, 'Unknown method [ %s ]' % method
        ucties = self.pred_to_uncertainty(preds)
        if self.model.use_custom_agg():
            ucties = self.model.aggregate_uncertainty(ucties, data_group)
        return ucties

    def current_uncertainty(self, data_group, algo='alpha_beta_max', method='hybrid'):
        if algo == 'alpha_beta_max':
            return self.uncertainty_alpha_beta(method, data_group=data_group)

    def recommend_acquisition_ids(self, num_recommend, algo, method='hybrid', agg_algo='alpha_beta_max'):
        if algo == 'active':
            ucties = self.current_uncertainty(method=method, data_group='pool', algo=agg_algo)
        elif algo == 'passive':
            ucties = [random.random() for i in range(self.model.get_pool_size())]
        else:
            assert False, 'Unknown algo [ %s ]' % algo
        neg_ucties = -np.array(ucties)
        pc = np.argpartition(neg_ucties, num_recommend)[:num_recommend]
        return pc, -neg_ucties[pc]

    def learn_by_cost(self, budget_size, algo, cost_ratio, fixed_fine_budget_ratio=None):
        budget_size = float(budget_size)
        # util_func = np.mean
        util_func = lambda x: np.mean([0.5] * 5 + [1.0 if v > 0.0 else 0.0 for v in x])
        if algo == 'fixed_fine_ratio':
            step_size_all = budget_size/(cost_ratio*fixed_fine_budget_ratio + (1 -fixed_fine_budget_ratio))
            budget_size_fine = fixed_fine_budget_ratio*cost_ratio*step_size_all
            budget_size_coarse = (1-fixed_fine_budget_ratio)*step_size_all
            step_size_fine = int(budget_size_fine / cost_ratio)
            step_size_coarse = int(budget_size_coarse)
            # handle the fraction that do not round up
            overhead = budget_size_fine/cost_ratio - step_size_fine
            if random.random() < overhead:
                step_size_fine += 1
            overhead = budget_size_coarse - step_size_coarse
            if random.random() < overhead:
                step_size_coarse += 1
            print >> sys.stderr, 'step_size_fine = %d step_size_coarse = %d' % (step_size_fine, step_size_coarse)
            self.go_allin_type(step_size=step_size_fine, method='fine', algo='active')
            self.go_allin_type(step_size=step_size_coarse, method='coarse', algo='active')
        elif algo == 'four_arm_bandit_uncertainty_sampling':
            step_size_fine = int(budget_size / cost_ratio)
            step_size_coarse = int(budget_size)
            # handle the fraction that do not round up
            overhead = 1.0 * budget_size/cost_ratio - step_size_fine
            if random.random() < overhead:
                step_size_fine += 1
            overhead = budget_size - step_size_coarse
            if random.random() < overhead:
                step_size_coarse += 1

            if self.last_played is None:
                self.go_allin_type(max(1, step_size_coarse), method='coarse', algo='active')
                self.last_played = 'coarse'
                self.last_reward = self.compute_reward()
            elif len(self.coarse_coarse_rewards) == 0:
                self.go_allin_type(max(1, step_size_coarse), method='coarse', algo='active')
                self.last_played = 'coarse'
                reward = self.compute_reward()
                self.coarse_coarse_rewards.append(math.log(reward)-math.log(self.last_reward))
                self.last_reward = reward
            elif len(self.coarse_fine_rewards) == 0:
                self.go_allin_type(max(1, step_size_fine), method='fine', algo='active')
                self.last_played = 'fine'
                reward = self.compute_reward()
                self.coarse_fine_rewards.append(math.log(reward)-math.log(self.last_reward))
                self.last_reward = reward
            elif len(self.fine_fine_rewards) == 0:
                self.go_allin_type(max(1, step_size_fine), method='fine', algo='active')
                self.last_played = 'fine'
                reward = self.compute_reward()
                self.fine_fine_rewards.append(math.log(reward)-math.log(self.last_reward))
                self.last_reward = reward
            elif len(self.fine_coarse_rewards) == 0:
                self.go_allin_type(max(1, step_size_coarse), method='coarse', algo='active')
                self.last_played = 'coarse'
                reward = self.compute_reward()
                self.fine_coarse_rewards.append(math.log(reward)-math.log(self.last_reward))
                self.last_reward = reward
            else:
                n_play_coarse_fine = len(self.coarse_fine_rewards)
                n_play_coarse_coarse = len(self.coarse_coarse_rewards)
                coarse_expect = (float(util_func(self.coarse_fine_rewards)) + math.sqrt(2.0* math.log(n_play_coarse_coarse+ n_play_coarse_fine)/ n_play_coarse_fine),
                                 float(util_func(self.coarse_coarse_rewards)) + math.sqrt(2.0* math.log(n_play_coarse_coarse+ n_play_coarse_fine)/ n_play_coarse_coarse))
                n_play_fine_fine = len(self.fine_fine_rewards)
                n_play_fine_coarse = len(self.fine_coarse_rewards)
                fine_expect= (float(util_func(self.fine_fine_rewards)) + math.sqrt(2.0 * math.log(n_play_fine_coarse + n_play_fine_fine) / n_play_fine_fine),
                              float(util_func(self.fine_coarse_rewards)) + math.sqrt(2.0 * math.log(n_play_fine_coarse + n_play_fine_fine) / n_play_fine_coarse))
                expect = (fine_expect, coarse_expect)
                nplay = ((n_play_fine_fine, n_play_fine_coarse), (n_play_coarse_fine, n_play_coarse_coarse))
                if self.last_played == 'coarse':
                    expect_fine, expect_coarse = coarse_expect
                else:
                    expect_fine, expect_coarse = fine_expect

                print >> sys.stderr, 'last play [ %s ] diff [ %.4f = %.4f - %.4f ] expect %s nplay %s step f/c %d %d' % \
                                     (self.last_played,
                                      expect_fine - expect_coarse,
                                      expect_fine, expect_coarse, expect, nplay, step_size_fine, step_size_coarse)
                if expect_fine > expect_coarse:
                    self.go_allin_type(step_size_fine, method='fine', algo='active')
                    reward = self.compute_reward()
                    if self.last_played == 'coarse':
                        self.coarse_fine_rewards.append(math.log(reward)-math.log(self.last_reward))
                    else:
                        self.fine_fine_rewards.append(math.log(reward)-math.log(self.last_reward))
                    self.last_played = 'fine'
                else:
                    self.go_allin_type(step_size_coarse, method='coarse', algo='active')
                    reward = self.compute_reward()
                    if self.last_played == 'coarse':
                        self.coarse_coarse_rewards.append(math.log(reward)-math.log(self.last_reward))
                    else:
                        self.fine_coarse_rewards.append(math.log(reward)-math.log(self.last_reward))
                    self.last_played = 'coarse'
                self.last_reward = reward

        elif algo == 'bandit_uncertainty_sampling':
            step_size_fine = int(budget_size / cost_ratio)
            step_size_coarse = int(budget_size)
            # handle the fraction that do not round up
            overhead = 1.0 * budget_size/cost_ratio - step_size_fine
            if random.random() < overhead:
                step_size_fine += 1
            overhead = budget_size - step_size_coarse
            if random.random() < overhead:
                step_size_coarse += 1

            if len(self.coarse_rewards) == 0:
                self.go_allin_type(max(1, step_size_coarse), method='coarse', algo='active')
                self.last_played = 'coarse'
                self.coarse_rewards.append(self.compute_reward())
            elif len(self.fine_rewards) == 0:
                self.go_allin_type(max(1, step_size_fine), method='fine', algo='active')
                self.fine_rewards.append(self.compute_reward())
            else:
                n_play_fine = len(self.fine_rewards)
                n_play_coarse = len(self.coarse_rewards)
                expect_fine = util_func(self.fine_rewards) + math.sqrt(2.0* math.log(n_play_coarse+ n_play_fine)/ n_play_fine)
                expect_coarse = util_func(self.coarse_rewards) + math.sqrt(2.0* math.log(n_play_coarse+ n_play_fine)/ n_play_coarse)
                print >> sys.stderr, 'diff [ %.4f = %.4f - %.4f ] nplay fine [ %d %d ] coarse [ %d %d ]' % \
                                     (expect_fine - expect_coarse, expect_fine, expect_coarse, step_size_fine, n_play_fine, step_size_coarse, n_play_coarse)
                if expect_fine > expect_coarse:
                    self.go_allin_type(step_size_fine, method='fine', algo='active')
                    self.fine_rewards.append(self.compute_reward())
                else:
                    self.go_allin_type(step_size_coarse, method='coarse', algo='active')
                    self.coarse_rewards.append(self.compute_reward())
        elif algo == 'bandit_e_greedy':
            step_size_fine = int(budget_size / cost_ratio)
            step_size_coarse = int(budget_size)
            # handle the fraction that do not round up
            overhead = 1.0 * budget_size/cost_ratio - step_size_fine
            if random.random() < overhead:
                step_size_fine += 1
            overhead = budget_size - step_size_coarse
            if random.random() < overhead:
                step_size_coarse += 1

            vv1 = [1.0 if v > 0 else 0.0 for v in self.coarse_fine_rewards]
            print >> sys.stderr, self.coarse_fine_rewards
            vv = [1.0, 0.0] * 5
            if len(vv1) > 0:
                #vv += list(np.random.choice(vv1, len(vv1), p=p))
                vv += vv1
            vv_inv = confint(sum(vv), len(vv), 0.1)
            vv_mean = np.average(vv_inv) - 0.5
            vv_width = (vv_inv[1] - vv_inv[0])/2
            vv_dist = 0.1
            if vv_mean * vv_mean - vv_width * vv_width < 0:
                vv_dist = max(0.1, 1.0 - abs(vv_mean / vv_width))
            expect_fine = vv_dist
            # check if lower than exploration prob
            if random.random() < expect_fine:
                if self.last_played == 'coarse':
                    next_play = 'fine'
                else:
                    next_play = 'coarse'
            else:
                if vv_mean > 0:
                    next_play = 'fine'
                else:
                    next_play = 'coarse'
            print >> sys.stderr, 'next [ %s ] diff [ %.4f ] nplay fine [ %d %d ] coarse [ %d %d ]' % \
                                 (next_play, expect_fine, step_size_fine, len(self.fine_rewards), step_size_coarse, len(self.coarse_rewards))
            if next_play == 'fine':
                self.go_allin_type(step_size_fine, method='fine', algo='active')
                self.fine_rewards.append(0)
            else:
                self.go_allin_type(step_size_coarse, method='coarse', algo='active')
                self.coarse_rewards.append(0)
            last_reward = self.compute_reward()
            if next_play != self.last_played:
                self.last_played = next_play
                if self.last_reward is not None:
                    if next_play == 'fine':
                        self.coarse_fine_rewards.append(last_reward - self.last_reward)
                    else:
                        self.coarse_fine_rewards.append(self.last_reward - last_reward)
            self.last_reward = last_reward
        else:
            assert False, 'Unknown algo [ %s ]' % algo

    def go_allin_type(self, step_size, method, algo):
        ids, ucties = self.recommend_acquisition_ids(step_size, method=method, algo=algo)
        self.model.acquire_example_ids(ids, example_type=method)
        self.model.fit()

    @staticmethod
    def score_max(raw_scores_list):
        raw_scores = [max(raw_scores_tuple) for raw_scores_tuple in zip(*raw_scores_list)]
        return raw_scores

    @staticmethod
    def pred_to_uncertainty(preds, agg=None):
        if agg == 'SUM':
            i = 1
        elif agg is not None:
            assert False, 'Unknown aggregation'
        ucties = [0.5 - abs(v - 0.5) for v in preds]
        return ucties

    @staticmethod
    def score_alpha_beta(raw_scores, agg=None):
        # alpha -> coarse weight
        # beta -> fine weight
        alpha = 0.5
        beta = 1.0 - alpha
        coarse_scores, fine_scores_list = raw_scores
        num_preds = len(coarse_scores)
        num_fines = len(fine_scores_list)
        if num_fines > 0:
            fine_scores = UncertaintySamplingController.score_max(fine_scores_list)
        else:
            # dummy scores
            fine_scores = coarse_scores
        # in case of fine only prediction
        if len(coarse_scores) == 0:
            coarse_scores = fine_scores
        hybrid_scores = [alpha*scores_tuple[0] + beta*scores_tuple[1]
                         for scores_tuple in zip(coarse_scores, fine_scores)]
        return hybrid_scores

    @staticmethod
    def unittest():
        controllers = list()
        controllers.append(IntervalModel.IntervalModel.unittest())
        for controller in controllers:
            preds = controller.score_alpha_beta(controller.model.predict_test_scores())
            assert min(preds) >= 0.0, 'negative uncertainty [%f]' % min(preds)
            assert max(preds) <= 1.0, 'prediction over 1.0 [%f]' % max(preds)
            ucties = controller.current_uncertainty()
            assert min(ucties) >= 0.0, 'negative uncertainty [%f]' % min(ucties)
            assert max(ucties) <= 0.5, 'uncertainty over 0.5 [%f]' % max(ucties)

if __name__ == '__main__':
    UncertaintySamplingController.unittest()