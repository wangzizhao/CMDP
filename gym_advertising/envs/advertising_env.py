import gym
from gym import spaces
import numpy as np

def AE_trans_prob(next_AE, all_AEs):
    """
    Compute trans prob for Advertising Effect using softmax
    Input:
    	next_AE: theoretical AE for next step
    	all_AEs: all possible value of AE
    """
    x = -20 * np.abs(all_AEs - next_AE)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def Choose_One(probs):
	rand = np.random.rand()
	for i, prob in enumerate(probs):
		rand -= prob
		if rand < 0:
			return i
	return None

class AdvertisingEnv(gym.Env):
	def __init__(self):
		pass

	def _set_params(self, nAdv, ndS, nA, alpha, beta, gamma, effects, period):
		assert nAdv == nA
		assert len(beta) == nAdv
		assert len(gamma) == nAdv
		assert len(effects) == nAdv
		assert period > 0

		self.action_space = spaces.Discrete(nA)
		self.observation_space = spaces.Tuple(tuple([spaces.Discrete(ndS) for _ in range(nAdv)]))
		self.nAdv = nAdv					# # of advertisement type
		self.ndS = ndS 						# # of states in each advertisement type
		self.nS = self.ndS ** self.nAdv		# # of states
		self.nA = nA 						# # of actions
		self.alpha = alpha					# decay rate of advertising effects for each type
		self.beta = beta					# user's sensitivity to ad of each type
		self.gamma = gamma					# sensitivity of visting rate to accumulated advertising effects
		self.effects = effects				# effects of an ad
		self.period = period				# how many ad are included in a period
		self.t = 0

	def Trans_Ob_2_State(self, ob):
		s = 0
		for idx, val in enumerate(reversed(ob)):
			s += int(np.round(val * (self.ndS - 1)) * self.ndS**idx)
		return s

	def Trans_State_2_Ob(self, s):
		ob = np.zeros(self.nAdv)
		for idx in range(self.nAdv):
			exponent = self.nAdv - 1 - idx
			ob[idx] = int(s/(self.ndS ** exponent))
			s -= ob[idx] * self.ndS ** exponent
			ob[idx] /= self.ndS - 1
		return ob

	def _get_obs(self):
		return tuple(self.AE)

	def reset(self):
		self.t = 0
		self.AE = np.zeros(self.nAdv, dtype = int)	# accumulated advertising effects
		return self._get_obs()

	def step(self, action):
		assert self.action_space.contains(action)

		reward = self.gamma[action] * self.AE[action]

		assert reward <= 1

		self.AE = np.multiply(self.AE, self.alpha)
		self.AE[action] += self.beta[action] * self.effects[action]

		all_AEs = np.arange(self.ndS)/(self.ndS - 1)
		for i in range(self.nAdv):
			prob = AE_trans_prob(self.AE[i], all_AEs)
			# print(self.AE[i], prob)
			idx = Choose_One(prob)
			if idx is None:
				print("cannot find a idx for prob:", prob)
			else:
				self.AE[i] = all_AEs[idx]
		self.AE = np.round(self.AE * (self.ndS - 1))/(self.ndS - 1)

		self.t += 1

		return self._get_obs(), reward, self.t == self.period, {}

	def Get_Trans_Prob_Rew(self):
		all_AEs = np.arange(self.ndS)/(self.ndS - 1)

		prob_table = np.zeros((self.nAdv, self.ndS, 2, self.ndS))
		for adv in range(self.nAdv):
			for s in range(self.ndS):
				for a in range(2):
					AE = s/(self.ndS - 1) * self.alpha[adv] + a * self.beta[adv] * self.effects[adv]
					prob_table[adv, s, a] = AE_trans_prob(AE, all_AEs)

		trans_prob = np.zeros((self.nS, self.nA, self.nS))
		rew = np.zeros((self.nS, self.nA))

		for s in range(self.nS):
			AE = self.Trans_State_2_Ob(s)

			for a in range(self.nA):
				prob_s_a = 1

				for i in range(self.nAdv):
					dS = int(np.round(AE[i] * (self.ndS - 1)))
					prob = prob_table[i, dS, int(a == i)]
					prob_s_a = np.array(np.dot(np.matrix(prob_s_a).T, np.matrix(prob))).reshape(-1)

				trans_prob[s, a] = prob_s_a
				rew[s, a] = self.gamma[a] * AE[a]

		return trans_prob, rew
