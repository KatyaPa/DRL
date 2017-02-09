#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from train_policy import BCNet

def rollout(args, policy_fn, bcnet):
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps =  env.spec.timestep_limit
        
        # import pdb; pdb.set_trace()
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            obs = env.reset()
            done = False
            steps = 0
            totalr = 0
            while not done:
                action = policy_fn(obs,bcnet)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr +=r
                steps += 1
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break                    
            returns.append(totalr)


        return  {   'observations': np.array(observations),
                    'actions': np.array(actions), 
                    'returns': np.array(returns),
                }


def policy_fn(inp,nn):
    # import pdb; pdb.set_trace()
    return nn.predict_for(inp)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', type=str, default='experts/Humanoid-v1.pkl')
    parser.add_argument('--envname', type=str, default='Humanoid-v1')
    parser.add_argument('--training_file', type=str, default='training_Humanoid-v1.pickle')
    parser.add_argument('--network_file', type=str, default='humanoid.net')
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--dagger', type=int, default=0,
        help='Number of dagger iterations')
    parser.add_argument('--hidden', type=int, default=100,
        help='Size of hidden layer')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--expert', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    bcnet = BCNet()
    with tf.Session():
        tf_util.initialize()

        expert_fn = load_policy.load_policy(args.expert_policy_file)
        if args.training:
            X, y = bcnet.get_train_data(args.training_file)
            bcnet.params['hidden']=args.hidden  
            bcnet.create_net()
            bcnet.net1.fit(bcnet.normalize_data(X),y)
            # import pdb; pdb.set_trace()            
            dagger_iters = []
            for i in range(args.dagger):
                # import pdb; pdb.set_trace()
                traj = rollout(args,policy_fn,bcnet)
                dagger_iters.append(traj)
                print 'New mean/std reward is: %f/%f' % (np.mean(traj['returns']), np.std(traj['returns']))
                new_input = traj['observations']
                new_output = np.squeeze(np.asarray(map(lambda x: expert_fn(x[None,:]),new_input)))
                X = np.concatenate((X,new_input))
                y = np.concatenate((y,new_output))
                print 'Total size of data: %d' % X.shape[0]
                bcnet = BCNet()
                bcnet.params['hidden']=args.hidden
                bcnet.params['inp_size']=X.shape[1]                  
                bcnet.params['outp_size']=y.shape[1]
                bcnet.create_net()
                bcnet.net1.fit(bcnet.normalize_data(X),y)
            bcnet.save_params_to(args.network_file)
            if args.dagger > 0:
                with open('%s_dagger.out' % args.envname,'w') as f:
                    pickle.dump(dagger_iters,f)
            return


        import gym
        env = gym.make(args.envname)
        print(env.action_space.shape)
        print(env.observation_space)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        if not args.expert: 
            bcnet.load_model(args.network_file)

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:                
                action = expert_fn(obs[None,:]) if args.expert else policy_fn(obs,bcnet)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)


        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions), 
                       'inp_size': env.observation_space.shape[0],
                       'outp_size': env.action_space.shape[0],
                       'mean_returns': np.mean(returns), 
                       'std_returns': np.std(returns),
                       }

        defname = 'training_%s.pickle' % args.envname if args.expert else 'test_%s.pickle' % args.envname
        fname = args.output_file if args.output_file is not None else defname
        with open(fname, 'w+') as f:
            pickle.dump(expert_data, f)


if __name__ == '__main__':
    main()
