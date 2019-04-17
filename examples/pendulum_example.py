import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(0)

from utils import rollout, policy


class myPendulum():
    def __init__(self):
        self.env = gym.make('Pendulum-v0').env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        high = np.array([np.pi, 1])
        self.env.state = np.random.uniform(low=-high, high=high)
        self.env.state = np.random.uniform(low=0, high=0.01*high) # only difference
        self.env.state[0] += -np.pi
        self.env.last_u = None
        return self.env._get_obs()

    def render(self):
        self.env.render()

        
env = myPendulum()

# Settings, are explained in the rest of the notebook
SUBS=3 # subsampling rate
T = 30 # number of timesteps (for planning, training and testing here)
J = 3 # rollouts before optimisation starts

max_action=2.0 # used by the controller, but really defined by the environment

# Reward function parameters
target = np.array([1.0, 0.0, 0.0])
weights = np.diag([2.0, 2.0, 0.3])

# Environment defined
m_init = np.reshape([-1.0, 0.0, 0.0], (1,3))
S_init = np.diag([0.01, 0.01, 0.01])

# Random rollouts
X,Y = rollout(env, None, timesteps=T, verbose=False, random=True, SUBS=SUBS, render=True)

for i in range(1,J):
    X_, Y_ = rollout(env, None, timesteps=T, verbose=False, random=True, SUBS=SUBS, render=True)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))
print(X)
state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10, max_action=max_action)
R = ExponentialReward(state_dim=state_dim, t=target, W=weights)
pilco = PILCO(X, Y, controller=controller, horizon=T, reward=R, m_init=m_init, S_init=S_init)

pilco.optimize_models(maxiter=100)
pilco.optimize_policy(maxiter=20)

# Rollout using the pilco controller
X_new, Y_new = rollout(env, pilco, timesteps=T, SUBS=SUBS, render=False)

for i,m in enumerate(pilco.mgpr.models):
    y_pred_test, var_pred_test = m.predict_y(X_new)
    plt.plot(range(len(y_pred_test)), y_pred_test, Y_new[:,i])
    plt.fill_between(range(len(y_pred_test)),
                       y_pred_test[:,0] - 2*np.sqrt(var_pred_test[:, 0]), 
                       y_pred_test[:,0] + 2*np.sqrt(var_pred_test[:, 0]), alpha=0.3)
    plt.show()
    
np.shape(var_pred_test)


from utils import predict_trajectory_wrapper
m_p = np.zeros((T, state_dim))
S_p = np.zeros((T, state_dim, state_dim))
for h in range(T):
    m_h, S_h, _ = predict_trajectory_wrapper(pilco, m_init, S_init, h)
    m_p[h,:], S_p[h,:,:] = m_h[:], S_h[:,:]
    

for i in range(state_dim):    
    plt.plot(range(T-1), m_p[0:T-1, i], X_new[1:T, i]) # can't use Y_new because it stores differences (Dx)
    plt.fill_between(range(T-1),
                     m_p[0:T-1, i] - 2*np.sqrt(S_p[0:T-1, i, i]),
                     m_p[0:T-1, i] + 2*np.sqrt(S_p[0:T-1, i, i]), alpha=0.2)
    plt.show()