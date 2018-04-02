import numpy as np
import time
import pickle

from KIWK_LR_Rmax import *
from CMDP_metric import *
from Simple_MDP import *
from Context import *

def Read_KWIK_LR_value(filename):
	if filename is None:
		return None
	data = Read_Data(filename)
	last_LR_history = data[-1][3]
	KWIK_LR_value = last_LR_history[0:4]


class CMDP:
	def __init__(self, MDP, episode = int(1e4), CtxModel = 0,
				 eps = 0.1, delta = 0.1,
				 b1_prob = 1, b2_prob = 1,
				 b1_rew = 1, b2_rew = 1,
				 keep_updating = False,
				 CMDP_init_value = None,
				 seed = None):
		self.MDP = MDP
		self.nC = nC = MDP.nC
		self.nS = nS = MDP.nS
		self.nA = nA = MDP.nA
		self.H = H = MDP.H

		self.episode = episode
		self.CtxModel = CtxModel

		self.eps = eps
		self.delta = delta
		alpha_s_prob = [b1_prob * eps**2 / nC**(3/2), 
						b2_prob * eps**2 / (np.sqrt(nC) * np.log(nC * 2**(nS) / delta)), 
						eps / (2 * np.sqrt(nC))]
		alpha_s_rew = [b1_rew * eps**2 / nC**(3/2), 
					   b2_rew * eps**2 / (np.sqrt(nC) * np.log(nC * 2**(nS) / delta)), 
					   eps / (2 * np.sqrt(nC))]
		self.alpha_s_prob = np.min(alpha_s_prob)
		self.alpha_s_rew = np.min(alpha_s_rew)
		self.converge_episode = nC**2 * H**4 * self.nS * nA / eps**5 \
							* np.log(1/delta * np.max([nC**2, self.nS**2 * (np.log(nC * self.nS * nA/delta))**2]))

		self.keep_updating = keep_updating

		KWIK_LR_value = None
		if CMDP_init_value is not None:
			self.c_t_history, self.LR_history, self.episode_history, self.trans_prob_history, \
				self.rew_histroy, self.policy_history, self.value_history = CMDP_init_value[2:9]
			KWIK_LR_value = CMDP_init_value[3][-1][0:4]
		self.LR_MDP = CMDP_KWIK_LR_RMAX(nC, self.nS, nA, self.alpha_s_prob, self.alpha_s_rew, KIWK_LR_init_value = KWIK_LR_value)
		
		self.file_name = "data\CMDP" + "\_nC_" + str(nC) \
									 + "_nA_" + str(nA) \
									 + "_H_" + str(H) \
									 + "_CtxModel_" + str(CtxModel) \
									 + "_" + time.strftime("%Y%m%d-%H%M%S") + ".p"

		self.episode_history = []

		self.seed(seed)

	def seed(self, seed = None):
		np.random.seed(seed)

	def Conduct(self):
		for epi in range(self.episode):

			if self.CtxModel == 0:
				c_t = generate_uniform_context(self.nC)
			# elif self.CtxModel == 1:
			# 	c_t = generate_uniform_context_with_redundant_dim(self.nC)
			# elif self.CtxModel == 2:
			# 	c_t = generate_uniform_context(self.nC)
			# elif self.CtxModel == 3:
			# 	c_t = generate_uniform_context(self.nC)

			self.MDP.Generate_CMDP(c_t)
			s = self.MDP.reset()
			
			goal_trans_prob, goal_rew = self.MDP.Get_Trans_Prob_Rew()
			LR_trans_prob, LR_rew = self.LR_MDP.GenertaeMDP(c_t)

			goal_policy, goal_value = Optimal_Control(goal_trans_prob, goal_rew, self.H)
			LR_policy, LR_value = Optimal_Control(LR_trans_prob, LR_rew, self.H)

			init_state_value_diff = goal_value[0, s] - LR_value[0, s]

			t = -1

			num_known = 0
			num_wrong_action = 0
			total_eps_rew = 0

			while True:
				t += 1

				if self.LR_MDP.Is_State_Known(c_t, s):
					num_known += 1
					action = LR_policy[t, s]
				else:
					action = self.LR_MDP.Unknown_action(c_t, s)

				s_next, reward, done, info = self.MDP.step(action)
				if not self.LR_MDP.Is_State_Known(c_t, s) or self.keep_updating:
					self.LR_MDP.Update(c_t, s, action, s_next, reward)

				if action != goal_policy[t,s]:
					num_wrong_action += 1
				total_eps_rew += reward

				s = s_next
				if done:
					break

			optimal_eps_rew = goal_value[0, s] * self.H
			epi_rew_diff = optimal_eps_rew - total_eps_rew
			
			self.episode_history.append([init_state_value_diff, num_wrong_action, epi_rew_diff])
			data_2_store = self.episode_history

			
			if epi > 0 and epi%100 == 0:
				print("episode:", epi)
				print("init_state_value_diff: {:5f}, num_known: {:d}, num_wrong_action: {:d}".format(init_state_value_diff, num_known, num_wrong_action))

			if epi > 0 and epi%int(1e5) == 0:
				with open(self.file_name, 'wb') as f:
					pickle.dump(data_2_store, f)

if __name__ == "__main__":
	MDP = simple_MDP(nC = 4, nA = 4, H = 4, e = 0.3)

	test1 = CMDP(MDP, episode = int(1e6), CtxModel = 0,
				eps = 0.1, delta = 0.1,
				b1_prob = 10, b2_prob = 10,
				b1_rew = 10, b2_rew = 10,
				keep_updating = False,
				CMDP_init_value = None,
				seed = None)
	test1.Conduct()