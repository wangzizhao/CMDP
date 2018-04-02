import numpy as np

def l1_norm(a):
	return np.sum(np.absolute(a))

class CMDP_KWIK_LR_RMAX:
	def __init__(self, C, S, A, alpha_s_prob, alpha_s_rew, KIWK_LR_init_value = None):
		self.alpha_s_prob = alpha_s_prob	# threshold for LR on transition prob
		self.alpha_s_rew = alpha_s_rew		# threshold for LR on rew
		self.d = C
		self.S = S
		self.A = A
		self.Initialize()
		if KIWK_LR_init_value is not None:
			self.Q_prob, self.W_prob, self.Q_rew, self.W_rew = KIWK_LR_init_value

	def Initialize(self):
		self.Q_prob = np.array([[np.identity(self.d) for _ in range(self.A)] for _ in range(self.S)])
		self.W_prob = np.zeros((self.S, self.A, self.d, self.S))
		self.Q_rew = np.array([[np.identity(self.d) for _ in range(self.A)] for _ in range(self.S)])
		self.W_rew = np.zeros((self.S, self.A, self.d))

	def GenertaeMDP(self, c):
		self.current_c = c
		self.prob = np.zeros((self.S, self.A, self.S))
		self.rew = np.zeros((self.S, self.A))
		self.known = np.zeros((self.S, self.A))
		for s in range(self.S):
			for a in range(self.A):
				self.prob[s, a], self.rew[s, a], self.known[s, a] = self.Predict(c, s, a)
		return self.prob, self.rew

	def Is_State_Action_Pair_Known(self, c, s, a):
		return l1_norm(np.dot(self.Q_prob[s,a], c)) <= self.alpha_s_prob and l1_norm(np.dot(self.Q_rew[s,a], c)) <= self.alpha_s_rew

	def Is_State_Known(self, c, s):
		assert np.array_equal(self.current_c, c)
		return np.sum(self.known[s]) == self.A

	def Unknown_action(self, c, s):
		assert np.array_equal(self.current_c, c)
		action = np.where(self.known[s] == False)[0]
		return np.random.choice(action)

	def Predict(self, c, s, a):
		"""
		Input:
			c: np.array with self.d elements
			s: int among 0, ..., self.S - 1
			a: int among 0, ..., self.A - 1
		Output:
			if known: return p(*|s,a) np.array with self.S element
			else: return None
		"""

		if self.Is_State_Action_Pair_Known(c, s, a):
			prob = self.Projection(np.dot(c, np.dot(self.Q_prob[s,a], self.W_prob[s,a])))
			rew = np.dot(c, np.dot(self.Q_rew[s,a], self.W_rew[s,a]))
			known = True
		else:
			prob = np.zeros(self.S)
			prob[s] = 1
			rew = 1
			known = False
		# if abs(np.sum(prob) - 1) >= 1e-5:
		# 	print("prob sum", np.sum(prob))
		# 	assert(abs(np.sum(prob) - 1) < 1e-5)
		return prob, rew, known

	def Update(self, c, s, a, s_next, rew):
		temp1 = np.dot(self.Q_prob[s,a], c)
		temp2 = (1 + np.dot(c, np.dot(self.Q_prob[s,a], c)))
		self.Q_prob[s,a] = self.Q_prob[s,a] - np.array(np.dot(np.matrix(temp1).T, np.matrix(temp1)))/temp2

		y = np.zeros(self.S)
		y[s_next] = 1
		self.W_prob[s,a] = self.W_prob[s,a] + np.array(np.dot(np.matrix(c).T, np.matrix(y)))

		temp1 = np.dot(self.Q_rew[s,a], c)
		temp2 = (1 + np.dot(c, np.dot(self.Q_rew[s,a], c)))
		self.Q_rew[s,a] = self.Q_rew[s,a] - np.array(np.dot(np.matrix(temp1).T, np.matrix(temp1)))/temp2

		self.W_rew[s,a] = self.W_rew[s,a] + rew * c

	def Projection(self, v):
		"""
		Porject prob from LR to space where sum = 1
		Algorithm 1 from Duchi et al 2008
		"""
		mu = np.sort(v)[::-1]
		for j in range(len(v)):
			temp = mu[j] - (np.sum(mu[:j+1]) - 1)/(j+1)
			if temp <= 0:
				break
		theta = (np.sum(mu[:j+1]) - 1)/(j+1)
		w = np.clip(v - theta, 0, 1)
		return w