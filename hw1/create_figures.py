import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import table


expert  = 'python run_test.py --expert --envname=%s --expert_policy_file=experts/%s.pkl --output_file=%s --num_rollouts=100'
bc      = 'python run_test.py --envname=%s --network_file=%s.net --output_file=%s --num_rollouts=100' 
dagger  = 'python run_test.py --network_file=Humanoid-v1_dagger.net --training --hidden=600 --dagger=15'


#def create_returns_file(fin, fout):
#    
#    obj = load_file(fin)
#
#    returns =  np.array(map(lambda d: [np.mean(d['returns']),np.std(d['returns'])], obj))
#
#    with open(fout ,'w') as f:
#        pickle.dump(returns,f)
#

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
    our_mean = round(obj['mean_returns'],3)
    our_std = round(obj['std_returns'],3)

    os.remove(ftmp)

    return [(expert_mean,expert_std), (our_mean, our_std)]
    

def create_table(RowLabels, cellText):
    
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
    plt.savefig('Table22.png',bbox_inches='tight')


def create_dagger_plot(returns):

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

    plt.savefig('Plot32.png',bbox_inches='tight')

    #plt.show()

def create_hyperparm_plot():
    return


def main():
    
    policies = ('Ant-v1','Humanoid-v1')
    
    cellText = []
    for policy in policies:
        cellText.append(create_row(policy))

    create_table(policies, cellText)
    create_hyperparm_plot()
    create_dagger_plot(cellText[1])

if __name__ == '__main__':
    main()


