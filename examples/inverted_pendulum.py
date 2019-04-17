import numpy as np
import matplotlib.pyplot as plt
import gym
import rl_environments
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
from tensorflow import logging
np.random.seed(0)

from utils import rollout, policy

config = tf.ConfigProto(device_count = {'GPU': 1})
config.gpu_options.allow_growth = True

# Reward function parameters: lin_pos[3] + ang_pos[3] + lin_vel[3] + ang_vel[3]
#                   x    y    z       r     p    q    x.   y.   z.   r.  p.    q.
target = np.array([0.0, 0.0, 0.4075, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
weights = np.diag([0.3, 0.3, 2.0, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2])

subs=2

m_init = np.random.randn(12)*0.01
S_init = m_init*0.0 + 0.02

with tf.Session(config=config, graph=tf.Graph()) as sess:
    env = gym.make('VrepBalanceBot2-v0')
    # Initial random rollouts to generate a dataset
    X,Y = rollout(env=env, pilco=None, random=True, timesteps=80, SUBS=subs, render=False)
    for i in range(1,12): #uniform action sampling
        X_, Y_ = rollout(env=env, pilco=None, random=True,  timesteps=80, SUBS=subs, render=False)
        X = np.vstack((X, X_)).astype(np.float64)
        Y = np.vstack((Y, Y_)).astype(np.float64)
    for i in range(1,24): #Gaussian/Normal distribution action sampling
        X_, Y_ = rollout(env=env, pilco=None, random="Normal",  timesteps=80, SUBS=subs, render=False)
        X = np.vstack((X, X_)).astype(np.float64)
        Y = np.vstack((Y, Y_)).astype(np.float64)
    for i in range(1,4): #No action sampling; u := 0
        X_, Y_ = rollout(env=env, pilco=None, random=None,  timesteps=80, SUBS=subs, render=False)
        X = np.vstack((X, X_)).astype(np.float64)
        Y = np.vstack((Y, Y_)).astype(np.float64)        


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=18, max_action=env.action_space.high)
    #controller = LinearController(state_dim=state_dim, control_dim=control_dim)
    #print(X)
    #pilco = PILCO(X, Y, controller=controller, horizon=40, num_induced_points=72)
    
    # Example of user provided reward function, setting a custom target state
    R = ExponentialReward(state_dim=state_dim, t=target)
    pilco = PILCO(X, Y, controller=controller, horizon=20, reward=R, 
                        num_induced_points=72, m_init=m_init)

    # Example of fixing a parameter, optional, for a linear controller only
    #pilco.controller.b = np.array([[0.0]])
    #pilco.controller.b.trainable = False
    plot_model = False # Plot model set to false until iteration 5 is reached
    for rollouts in range(20):
        print("Optimizing models") 
        pilco.optimize_models()
        print("Optimizing controller")
        pilco.optimize_policy(maxiter=20)
        #import pdb; pdb.set_trace()
        X_new, Y_new = rollout(env=env, pilco=pilco, timesteps=120, SUBS=subs, render=False)
        print("No of ops:", len(tf.get_default_graph().get_operations()))
        # Update dataset
        X = np.vstack((X, X_new)).astype(np.float64)
        Y = np.vstack((Y, Y_new)).astype(np.float64)
        pilco.mgpr.set_XY(X, Y)
        
        if rollouts > 16:
            plot_model = True

        if plot_model:
            #Plot the model
            var_iter = 0
            col_num = 3
            row_num = int(state_dim / col_num)
            fig_s, ax_s = plt.subplots(row_num, col_num, sharex='all', sharey='row')
            fig_s.suptitle('Predicted Dynamics, Single Step', fontsize=16) 
            ax_s[0,0].set_title('Axis X')
            ax_s[0,1].set_title('Axis Y')
            ax_s[0,2].set_title('Axis Z')
            ax_s[0,0].set_ylabel('Linear Pos. [m]')
            ax_s[1,0].set_ylabel('Angular Pos. [rad]')
            ax_s[2,0].set_ylabel('Linear Vel. [m/s]')
            ax_s[3,0].set_ylabel('Angular Vel. [rad/s]')
            ax_s[3,1].set_xlabel('Timesteps @10hz')

            for i,m in enumerate(pilco.mgpr.models): #show the single-step Prediction
                var_iter_x = var_iter%row_num
                var_iter_y = int((var_iter - var_iter_x)/row_num)

                y_pred_test, var_pred_test = m.predict_y(X_new)
                ## NEEDS TESTING
                ax_s[var_iter_x,var_iter_y].grid(True)
                ax_s[var_iter_x,var_iter_y].plot(range(len(y_pred_test)), y_pred_test, Y_new[:,i])
                ax_s[var_iter_x,var_iter_y].fill_between(range(len(y_pred_test)),
                                y_pred_test[:,0] - 2*np.sqrt(var_pred_test[:, 0]), 
                                y_pred_test[:,0] + 2*np.sqrt(var_pred_test[:, 0]), alpha=0.3)
                ## New plot above
                
                #plt.plot(range(len(y_pred_test)), y_pred_test, Y_new[:,i])
                #plt.fill_between(range(len(y_pred_test)),
                #                   y_pred_test[:,0] - 2*np.sqrt(var_pred_test[:, 0]), 
                #                   y_pred_test[:,0] + 2*np.sqrt(var_pred_test[:, 0]), alpha=0.3)
                var_iter += 1
            #plt.show() #move the show command to outside the loop

            from utils import predict_trajectory_wrapper #show the multi-step prediction
            T=20
            #init as the first measured position, because X := [state,action] array, remove the action dim
            m_init = np.round(np.reshape(X_new[0,:-control_dim], (1,-1)), decimals=2)
            S_init = np.diag(m_init[0,:] + 0.02)

            m_p = np.zeros((T, state_dim))
            S_p = np.zeros((T, state_dim, state_dim))
            for h in range(T):
                m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
                m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
                    
            var_iter = 0
            col_num = 3
            row_num = int(state_dim / col_num)
            fig_m, ax_m = plt.subplots(row_num, col_num, sharex='all', sharey='row')
            fig_m.suptitle('Predicted Dynamics, Multiple Step', fontsize=16) 
            ax_m[0,0].set_title('Axis X')
            ax_m[0,1].set_title('Axis Y')
            ax_m[0,2].set_title('Axis Z')
            ax_m[0,0].set_ylabel('Linear Pos. [m]')
            ax_m[1,0].set_ylabel('Angular Pos. [rad]')
            ax_m[2,0].set_ylabel('Linear Vel. [m/s]')
            ax_m[3,0].set_ylabel('Angular Vel. [rad/s]')
            ax_m[3,1].set_xlabel('Timesteps @10hz')
    
            for i in range(state_dim):
                var_iter_x = var_iter%row_num
                var_iter_y = int((var_iter - var_iter_x)/row_num)

                ax_m[var_iter_x,var_iter_y].grid(True)
                ax_m[var_iter_x,var_iter_y].plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
                ax_m[var_iter_x,var_iter_y].fill_between(range(T-1),
                                    m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
                                    m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)

                var_iter += 1
            plt.show()