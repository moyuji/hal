from subprocess import call
import ujson as json
import sys, os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
def update_from_HDFS():
    print >> sys.stderr, 'Downloads started'
    dali_get = ["/usr/local/linkedin/bin/dali", "fs", "-copyToLocal"]
    webfs = "dali://holdem"
    call(["rm", "-rf", "../build/results"])
    call(dali_get + [webfs + "/user/ymo/hal/results", '../build/'])
    print >> sys.stderr, 'Downloads finished'

def get_raw_results(halid, task_lab):
    results_paths = ['../build/results/', 'artifact/results/']
    results = dict()
    for root_path in results_paths:
        path = root_path + task_lab + '/' + halid
        path = os.path.realpath(path)
        if not os.path.exists(path):
            continue
        for filename in os.listdir(path):
            if not filename.startswith('part-'):
                continue
            if filename.endswith('.gz'):
                fp = gzip.open(path + '/' + filename)
            else:
                fp = open(path + '/' + filename)
            for line in fp:
                x = json.loads(line)
                for k, v in x.items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v)
    df = pd.DataFrame(results)
    return df

def get_results(halid, task_lab, view_angle = None):
    df = get_raw_results(halid, task_lab)
    row_total = df.shape[0]
    col_total = df.shape[1]
    if task_lab == 'manyTypes':
        df['config'] = [a + '/' + b for (a,b) in zip(df['algo'], df['method'])]
        df = df.groupby(['config', 'round']).agg({'prauc': np.mean})
    elif task_lab == 'dynamicRatio' and view_angle is None:
        df['config'] = ['FFR[%.1f]' % b if a == 'fixed_fine_ratio' else 'BANDIT' for (a,b) in zip(df['algo'], df['fine_ratio'])]
        df['fine_cost'] = [str(a) for a in df['fine_cost']]
        df_mixed = df.copy(deep=True)
        df['fine_cost'] = ['mixed' for a in df['fine_cost']]
        df = pd.concat([df, df_mixed]).groupby(['fine_cost', 'config', 'round']).agg({'prauc': np.mean})
    elif task_lab == 'dynamicRatio' and view_angle == 'nobandit':
        df['config'] = ['FFR[%.1f]' % b if a == 'fixed_fine_ratio' else 'BANDIT' for (a,b) in zip(df['algo'], df['fine_ratio'])]
        df['fine_cost'] = [str(a) for a in df['fine_cost']]
        df = df[df['config'] != 'BANDIT'].groupby(['fine_cost', 'config', 'round']).agg({'prauc': np.mean})
    elif task_lab == 'dynamicRatio':
        df['config'] = ['FFR[%.1f]' % b if a == 'fixed_fine_ratio' else 'BANDIT' for (a,b) in zip(df['algo'], df['fine_ratio'])]
    print 'Processed result [ %s %s ] from [ %d rows %d repeats %d columns ] into [ %d rows ]' %\
          (halid, task_lab, row_total, row_total/df.shape[0], col_total, df.shape[0])
    return df

def show_figure(df, legend_loc=4, xlabel='Iteration'):
    for key in df.index.levels[0]:
        x = df.index.levels[1]
        y = df.loc[key]
        plt.plot(x, y)
    plt.legend(df.index.levels[0], loc=legend_loc)
    plt.xlabel(xlabel)
    plt.ylabel('PR-AUC')

cards = ['interval', 'RCV1', 'richmond']
#cards = ['RCV1', 'richmond']

num_cards = len(cards)

def plotManyTypes():
    sns.set_palette(sns.color_palette("hls", 4))
    plt.figure(figsize=(26, 6), dpi=300)
    for i, task_model in enumerate(cards):
        plt.subplot(1, num_cards, i+1)
        df = get_results(task_model, 'manyTypes')
        plt.title('%s: Comparing Fine/Coarse Active/Passive' % task_model)
        show_figure(df)
    plt.savefig('figure/draft.png')

    for i, task_model in enumerate(cards):
        plt.figure(figsize=(7, 5), dpi=300)
        df = get_results(task_model, 'manyTypes')
        show_figure(df)
        plt.savefig('figure/draft-%s.png' % task_model)

def plotDynamicRatio():
    sns.set_palette(sns.color_palette(["#34495e"] + sns.color_palette("hls", 15)))
    for j, task_model in enumerate(cards):
        plt.figure(figsize=(28,14), dpi=300)
        z = get_results(task_model, 'dynamicRatio')
        for i, v in enumerate(z.index.levels[0]):
            plt.subplot(3,3,i+1)
            plt.title('%s: Ratio at %s' % (task_model, v))
            show_figure(z.loc[v])
        plt.savefig('figure/%s-bandit.png' % task_model)
        plt.close()

    for j, task_model in enumerate(cards):
        sns.set_palette(sns.color_palette(["#34495e"] + sns.color_palette("hls", 15)))
        z = get_results(task_model, 'dynamicRatio')
        for i, v in enumerate(z.index.levels[0]):
            plt.figure(figsize=(7, 5), dpi=300)
            #plt.title('%s: Ratio at %s' % (task_model, v))
            show_figure(z.loc[v])
            plt.savefig('figure/%s-%s-bandit.png' % (task_model, v))
            plt.close()

        z = get_results(task_model, 'dynamicRatio', 'nobandit')
        sns.set_palette(sns.color_palette(sns.color_palette("hls", 15)))
        for i, v in enumerate(z.index.levels[0]):
            plt.figure(figsize=(7, 5), dpi=300)
            #plt.title('%s: Ratio at %s' % (task_model, v))
            show_figure(z.loc[v])
            plt.savefig('figure/%s-%s-nobandit.png' % (task_model, v))
            plt.close()


def plotDynamicRatio2():
    sns.set_palette(sns.color_palette(["#34495e"] + sns.color_palette("hls", 15)))
    round_map = {'interval': 499, 'RCV1': 99, 'richmond': 39}
    plt.figure(figsize=(26, 6), dpi=300)
    for j, task_model in enumerate(cards):
        plt.subplot(1, num_cards, j+1)
        z = get_results(task_model, 'dynamicRatio', 'Raw')
        z = z[z['round'] == round_map[task_model]]
        z['log_fine_cost'] = [math.log(a) for a in z['fine_cost']]
        z = z.groupby(['config', 'log_fine_cost']).agg({'prauc': np.mean})
        plt.title('%s: Bandit at iteration %d' % (task_model, round_map[task_model]))
        show_figure(z, xlabel='Log Fine Cost')
    plt.savefig('figure/curve.png')

    for j, task_model in enumerate(cards):
        plt.figure(figsize=(7, 4.5), dpi=600)
        z = get_results(task_model, 'dynamicRatio', 'Raw')
        z = z[z['round'] == round_map[task_model]]
        z['log_fine_cost'] = [math.log(a) for a in z['fine_cost']]
        z = z.groupby(['config', 'log_fine_cost']).agg({'prauc': np.mean})
        show_figure(z, xlabel='Log fine cost')
        plt.savefig('figure/curve-%s.png' % task_model)

def main():
    #update_from_HDFS()
    plotManyTypes()
    plotDynamicRatio()
    plotDynamicRatio2()

if __name__ == '__main__':
    main()
