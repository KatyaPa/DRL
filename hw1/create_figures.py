import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.table import table


fout    = 'tmp.pickle'
#nroll   = 20
expert  = 'python run_test.py --expert --envname=%s --expert_policy_file=experts/%s.pkl --num_rollouts=%d --output_file='+fout
our     = 'python run_test.py  --envname=%s --network_file=%s.net --num_rollouts=%d --output_file='+fout      


def load_fout():

    with open(fout,'r') as f:
        return pickle.load(f)


def create_row(policy, nroll):

    print expert % (policy,policy,nroll)    
    os.system(expert % (policy,policy,nroll))
    obj = load_fout()
    expert_mean = round(obj['mean_returns'],3)
    expert_std = round(obj['std_returns'],3)

    print our % (policy,policy,nroll)
    os.system(our % (policy,policy,nroll))
    obj = load_fout()
    our_mean = round(obj['mean_returns'],3)
    our_std = round(obj['std_returns'],3)

    os.remove(fout)

    return [(expert_mean,expert_std), (our_mean, our_std)]
    

def create_table(nroll):
    
    ColLabels = ('Expert\n'+'(mean,std)', 'Ours\n'+'(mean,std)')
    
    RowLabels = ('Walker2d-v1','Humanoid-v1')
    
    cellText = []
    for policy in RowLabels:
        cellText.append(create_row(policy,nroll))

    ncols = len(ColLabels)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    the_table =ax.table(cellText=cellText,
                        colWidths=[.2]*ncols,
                        rowLabels= RowLabels, 
                        colLabels=ColLabels, 
                        loc='center') 
                        
    the_table.auto_set_font_size(True)
    the_table.scale(1.5,4.)
    plt.savefig('Table22.png',bbox_inches='tight')




def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of roll outs')

    args = parser.parse_args()
   
    create_table(args.num_rollouts)
    

if __name__ == '__main__':
    main()


