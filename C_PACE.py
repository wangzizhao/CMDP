import numpy as np
import time
import pickle

from annoy import AnnoyIndex

from Simple_MDP import *
from Context import *

class C_PACE:
	def __init__(self, MDP, episode = int(1e4), CtxModel = 0,
				k = 20, L_Q = 5, eps_d = 0.25,
				seed = None):
		self.MDP = MDP
		self.nC = nC = MDP.nC
		self.nS = nS = MDP.nS
		self.nA = nA = MDP.nA
		self.H = H = MDP.H

		self.episode = episode
		self.CtxModel = CtxModel

		self.k = k
		self.L_Q = L_Q
		self.eps_d = eps_d

		self.n_trees = 50
		self.c_list = [[[] for _ in range(nA)] for _ in range(nS)]			# the list store c for each s-a pair
		self.r_s_next_list = [[[] for _ in range(nA)] for _ in range(nS)]	# the list storing (r, s') for each s-a pair

		self.file_name = "data\C_PACE" + "\_nC_" + str(nC) \
									+ "_nA_" + str(nA) \
									+ "_H_" + str(H) \
									+ "_CtxModel_" + str(CtxModel) \
									+ "_" + time.strftime("%Y%m%d-%H%M%S") + ".p"
		self.episode_history = []
		self.seed(seed)
		self.Init_Annoy()

	def seed(self, seed = None):
		np.random.seed(seed)

	def Init_Annoy(self):
		nS, nA = self.nS, self.nA
		self.annoy_idxs = [[AnnoyIndex(self.nC, metric = 'euclidean') for _ in range(nA)] for _ in range(nS)]
		for s in range(nS):
			for a in range(nA):
				self.annoy_idxs[s][a].build(50)

	def Update_Annoy(self, s, a):
		self.annoy_idxs[s][a] = AnnoyIndex(self.nC, metric = 'euclidean')
		for i, c in enumerate(self.c_list[s][a]):
			self.annoy_idxs[s][a].add_item(i, c)
		self.annoy_idxs[s][a].build(50)


	def Optimal_Control(self, c_t):
		nS, nA, H, k = self.nS, self.nA, self.H, self.k

		self.is_known = np.zeros((nS, nA), dtype = bool)
		self.k_neighbor_idxs = np.zeros((nS, nA, k), dtype = int)

		# find k neighbors for c_t in c_list for each s-a pair
		dists = [[[0.0 for _ in range(k)] for _ in range(nA)] for _ in range(nS)]	
		for s in range(nS):
			for a in range(nA):
				len_c = len(self.c_list[s][a])
				if len_c < k:
					self.k_neighbor_idxs[s, a] = -np.ones(k, dtype = int)
					self.k_neighbor_idxs[s, a, 0:len_c] = np.arange(len_c, dtype = int)
				else:
					idxs = self.annoy_idxs[s][a].get_nns_by_vector(c_t, k)
					self.k_neighbor_idxs[s, a] = idxs
				self.is_known[s, a] = True
				for i in range(k):
					i_th_nn_idx = self.k_neighbor_idxs[s, a, i]
					if i_th_nn_idx != -1:
						i_th_nn = self.annoy_idxs[s][a].get_item_vector(i_th_nn_idx)
						dists[s][a][i] = np.linalg.norm(c_t - i_th_nn)
						if dists[s][a][i] > self.eps_d/self.L_Q:
							self.is_known[s, a] = False
					else:
						self.is_known[s, a] = False

		# find policy
		policy = np.zeros((H, nS), dtype = int)
		value = np.zeros((H, nS))
		value_t = np.zeros(nS)
		new_value_t = np.zeros(nS)

		for i, t in zip(range(H), reversed(range(H))):
			Q_max = i + 1
			for s in range(nS):
				Q_s_a = []
				for a in range(nA):
					Q_sum = 0
					for i in range(k):
						if self.k_neighbor_idxs[s, a, i] != -1:
							i_th_nn_idx = self.k_neighbor_idxs[s, a, i]
							r, s_next = self.r_s_next_list[s][a][i_th_nn_idx]
							x_i = r + value_t[s_next]
							Q_sum += np.minimum(Q_max, x_i + self.L_Q * dists[s][a][i])
						else:
							Q_sum += Q_max
					Q_s_a.append(Q_sum/k)
				new_value_t[s] = np.max(Q_s_a)
				policy[t, s] = np.argmax(Q_s_a)
			value_t = np.array(new_value_t)
			value[t] = value_t

		return policy, value

	def Conduct(self):
		for epi in range(self.episode):

			# c_t = generate_uniform_context(self.nC)

			if self.CtxModel == 0:
				c_t = generate_uniform_context(self.nC)
			# elif self.CtxModels == 1:
			# 	c_t = generate_uniform_context_with_redundant_dim(self.nC)
			elif self.CtxModels == 2:
				c_t = generate_binodal_context(self.nC, epi, self.episode)
			# elif self.CtxModels == 3:
			# 	c_t = generate_uniform_context(self.nC)

			self.MDP.Generate_CMDP(c_t)
			s_0 = s = self.MDP.reset()

			goal_trans_prob, goal_rew = self.MDP.Get_Trans_Prob_Rew()
			goal_policy, goal_value = Optimal_Control(goal_trans_prob, goal_rew, self.H)
			policy, value = self.Optimal_Control(c_t)

			init_state_value_diff = goal_value[0, s] * self.H - value[0, s]

			num_known = self.H
			num_wrong_action = 0
			total_eps_rew = 0

			t = -1
			while True:
				t += 1

				a = policy[t, s]

				s_next, reward, done, info = self.MDP.step(a)

				if not self.is_known[s, a]:
					num_known -= 1
					self.c_list[s][a].append(c_t)
					self.r_s_next_list[s][a].append([reward, s_next])
					self.Update_Annoy(s, a)
					policy, _ = self.Optimal_Control(c_t)

				if a != goal_policy[t,s]:
					num_wrong_action += 1
				total_eps_rew += reward

				s = s_next
				if done:
					break

			epi_rew_diff = goal_value[0, s_0] - total_eps_rew
			self.episode_history.append([init_state_value_diff, num_wrong_action, epi_rew_diff])
			data_2_store = self.episode_history

			if epi%100 == 0 and epi > 0:
				print(epi)
				print("init_state_value_diff: {:5f}, num_known: {:d}, num_wrong_action: {:d}".format(init_state_value_diff, num_known, num_wrong_action))

			if epi > 0 and epi%int(2e3) == 0:
				with open(self.file_name, 'wb') as f:
					pickle.dump(data_2_store, f)

if __name__ == "__main__":
	MDP = simple_MDP(nC = 4, nA = 4, H = 4, e = 0.3)

	test1 = C_PACE(MDP, episode = int(2e5),
				k = 100, L_Q = 10, eps_d = 0.2)
	test1.Conduct()
