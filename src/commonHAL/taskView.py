import sys
import json
import argparse
import IntervalModel
import RCV1Model
import RichmondModel
import gc

def run_from_tasks(f_input = sys.stdin):
    for line in f_input:
        gc.collect()
        param = json.loads(line)
        task_model = param['model']
        task_lab = param['lab']
        task_iteration = param['iteration']
        task_repeat = param['repeat']
        if task_model == 'interval':
            factory = IntervalModel
        elif task_model == 'RCV1':
            factory = RCV1Model
        elif task_model == 'richmond':
            factory = RichmondModel
        else:
            assert False, 'Unknown model type [ %s ]' % task_model
        if task_lab == 'manyTypes':
            controller = factory.set_up()
            task_method = param['method']
            task_algo = param['algo']
            task_budget = param['budget']
            for task_round in range(task_iteration):
                metric = controller.current_metric()
                print >> sys.stderr, 'repeat %d [ %s  %s ] iteration %d = %.4f' % (task_repeat, task_method, task_algo, task_round, metric)
                param['prauc'] = metric
                param['round'] = task_round
                print json.dumps(param)
                pcs, ucties = controller.recommend_acquisition_ids(task_budget, algo=task_algo, method=task_method)
                controller.model.acquire_example_ids(pcs, task_method)
                controller.model.fit()

        elif task_lab == 'dynamicRatio':
            task_algo = param['algo']
            task_fine_cost = param['fine_cost']
            task_fine_ratio = param['fine_ratio']
            task_budget = param['budget']
            if task_algo == 'fixed_fine_ratio':
                controller = factory.set_up()
            elif task_algo == 'bandit_uncertainty_sampling':
                controller = factory.set_up(pond_enabled=True)
            elif task_algo == 'four_arm_bandit_uncertainty_sampling':
                controller = factory.set_up(pond_enabled=True)
            elif task_algo == 'bandit_e_greedy':
                controller = factory.set_up(pond_enabled=True)
            else:
                assert False, 'Unknown algo [ %s ]' % task_algo
            for task_round in range(task_iteration):
                metric = controller.current_metric()
                print >> sys.stderr, 'repeat %d [ cost %.2f %s ratio %.2f ] iteration %d = %.3f' % \
                                     (task_repeat, task_fine_cost, task_algo, task_fine_ratio, task_round, metric)
                param['prauc'] = metric
                param['round'] = task_round
                print json.dumps(param)
                controller.learn_by_cost(budget_size=task_budget,
                                         algo=task_algo,
                                         cost_ratio=task_fine_cost,
                                         fixed_fine_budget_ratio=task_fine_ratio)

        else:
            assert False, 'Unkown lab type [ %s ]' % task_lab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t')
    parser.add_argument('-f')
    args = parser.parse_args()
    if args.t == 'tasks':
        run_from_tasks(sys.stdin)
    if args.t == 'unittest':
        with open(args.f) as f_input:
            run_from_tasks(f_input)