#!/usr/bin/env python

"""
Code generate plots.
Example usage:
    python create_figures.py --table --hyperparam --dagger --show

"""



import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import table


expert      = 'python run_test.py --expert --envname=%s --expert_policy_file=experts/%s.pkl --output_file=%s --num_rollouts=100'
bc          = 'python run_test.py --envname=%s --network_file=%s.net --output_file=%s --num_rollout=100' 

dagger      = 'python run_test.py --network_file=Humanoid-v1_dagger.net --training --hidden=150 --dagger=10'

train_ant   = 'python run_test.py --envname=Ant-v1 --network_file=Ant-v1_%d.net --training --epochs=%d --training_file=training_Ant-v1.pickle'
test_ant    = 'python run_test.py --envname=Ant-v1 --network_file=Ant-v1_%d.net --output_file=%s --num_rollouts=100'



def load_file(fin):

    print("Loading file")
    with open(fin,'r') as f:
        obj = pickle.load(f)
    print('Finished loading file')
    return obj


def create_row(policy):

    ftmp    = 'tmp.pickle'

    print expert % (policy,policy,ftmp)    
    os.system(expert % (policy,policy,ftmp))
    obj = load_file(ftmp)
    expert_mean = round(obj['mean_returns'],3)
    expert_std = round(obj['std_returns'],3)

    print bc % (policy,policy,ftmp)
    os.system(bc % (policy,policy,ftmp))
    obj = load_file(ftmp)
    bc_mean = round(obj['mean_returns'],3)
    bc_std = round(obj['std_returns'],3)

    os.remove(ftmp)

    return [(expert_mean,expert_std), (bc_mean, bc_std)]
    
def run_test_epochs(epoch):

    ftmp  = 'tmp.pickle'
    print train_ant %(epoch, epoch)
    os.system(train_ant %(epoch, epoch))
    print test_ant %(epoch, ftmp)
    os.system(test_ant %(epoch, ftmp))
    obj = load_file(ftmp)
    bc_mean = round(obj['mean_returns'],3)
    bc_std = round(obj['std_returns'],3)

    os.remove(ftmp)
    
    return (bc_mean, bc_std)


def create_table(args):

    RowLabels = ('Ant-v1','Humanoid-v1')
    
    cellText = []
    for policy in RowLabels:
        cellText.append(create_row(policy))

    ColLabels = ('Expert\n'+'(mean,std)', 'BC\n'+'(mean,std)')
    
    ncols = len(ColLabels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    the_table = ax.table(cellText=cellText,
                        colWidths=[.2]*ncols,
                        rowLabels= RowLabels, 
                        colLabels=ColLabels, 
                        loc='center') 
                        
    the_table.auto_set_font_size(True)
    the_table.scale(1.5,4.)

    if args.save:
        plt.savefig(args.table_file,bbox_inches='tight')

    if args.show:
        plt.show()


def create_hyperparm_plot(args):

    fig = plt.figure()    

    hyper_return = []
    x = [30, 50, 70, 100, 150, 200, 300, 400]
    for epochs in x:
        hyper_return.append(run_test_epochs(epochs))
    #import pdb; pdb.set_trace()
    hyper_return = np.array(hyper_return)

    y_hyper   = hyper_return[:,0]
    e_hyper   = hyper_return[:,1]
    plt.errorbar(x, y_hyper, yerr=e_hyper, linestyle='-', marker='o', ecolor='g')

    if args.save:
        plt.savefig(args.hyperparm_file,bbox_inches='tight')

    if args.show:
        plt.show()


def create_dagger_plot(args):

    returns = create_row('Humanoid-v1')

    os.system(dagger)
    fin     = 'Humanoid-v1_dagger.out'
    dagger_returns = load_file(fin)
    
    dagger_returns =  np.array(map(lambda d: [np.mean(d),np.std(d)], dagger_returns))

    fig = plt.figure()

    n_daggers  = len(dagger_returns)
    x           = np.arange(n_daggers)+1

    y_dagger   = dagger_returns[:,0]
    e_dagger   = dagger_returns[:,1]
    plt.errorbar(x, y_dagger, yerr=e_dagger, linestyle='-', marker='o', ecolor='g')

    y_expert = [returns[0][0]]*n_daggers
    e_expert = [returns[0][1]]*n_daggers
    plt.errorbar(x,y_expert, yerr =e_expert  , linestyle='--', marker='^', color='k')

    y_bc = [returns[1][0]]*n_daggers
    e_bc = [returns[1][1]]*n_daggers
    plt.errorbar(x,y_bc, yerr =e_bc  , linestyle='--', marker='^', color='r')

    if args.save:
        plt.savefig(args.dagger_file,bbox_inches='tight')

    if args.show:
        plt.show()


def main():
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--table_file', type=str, default='Table22.png')
    parser.add_argument('--hyperparm_file', type=str, default='Plot23.png')
    parser.add_argument('--dagger_file', type=str, default='Plot32.png')

    parser.add_argument('--table', action='store_true')
    parser.add_argument('--hyperparm', action='store_true')
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()


    if args.table:
        create_table(args)
    if args.hyperparm:
        create_hyperparm_plot(args)
    if args.dagger:
        create_dagger_plot(args)

    create_dagger_plot_tmp()

if __name__ == '__main__':
    main()


