import numpy as np

def Optimal_Control(P, R, H):
	"""
	P.shape = (nS, nA, nS)
	R.shape = (nS, nA)
	H: length of an episode
	"""
	nS, nA = R.shape
	policy = np.zeros((H, nS), dtype = int)
	value = np.zeros((H, nS))
	value_t = np.zeros(nS)
	new_value_t = np.zeros(nS)
	for i, t in zip(range(H), reversed(range(H))):
		for s in range(nS):
			q = []
			for a in range(nA):
				q.append((R[s, a] + i * np.dot(P[s,a], value_t))/(i + 1))	# Q(s, a)
			new_value_t[s] = np.max(q)										# V(s) = max Q(s, a)
			policy[t, s] = np.argmax(q)
		value_t = np.array(new_value_t)
		value[t] = value_t

	return policy, value

def Choose_One(probs):
	rand = np.random.rand()
	for i, prob in enumerate(probs):
		rand -= prob
		if rand < 0:
			return i
	return None

class simple_MDP:
	def __init__(self, nC = 6, nA = 4, H = 2, e = 0.3):
		assert(nA >= nC)
		nS = nC + 2
		self.nC = nC
		self.nS = nS
		self.nA = nA
		self.H = H
		self.T = np.zeros((nC, nS, nA, nS))
		self.R = np.zeros((nC, nS, nA))
		self.e = e
		self.Initialize_T_R()
		self.Generate_CMDP(np.ones(nC)/nC)

	def Initialize_T_R(self):
		temp0 = np.ones(self.nS)/(self.nS - 3)
		temp0[0] = temp0[-1] = temp0[-2] = 0
		temp1 = np.zeros((self.nS))
		temp1[-2], temp1[-1] = 0.5, 0.5
		temp2 = np.zeros((self.nS))
		temp2[-2], temp2[-1] = 0.5 + self.e, 0.5 - self.e
		for c in range(self.nC):
			# p(i|0, a) = 1/n (i: 1~n)
			for a in range(self.nA):
				self.T[c, 0, a] = temp0
			# p(+|i, a) = 1/2 + e_i(a)
			# p(-|i, a) = 1/2 - e_i(a)
			for s in range(1, self.nS - 2):
				for a in range(self.nA):
					self.T[c, s, a] = temp1
				self.T[c, s, 0] = temp2
				self.T[c, s, c] = temp2
			# p(+|+, a) = 1
			# r(+, a) = 1
			# p(-|-, a) = 1
			# r(-, a) = 0
			for a in range(self.nA):
				self.T[c, -2, a, -2] = 1
				self.T[c, -1, a, -1] = 1
				self.R[c, -2, a] = 1
		# print(self.T[1])
		# print(self.R[0])

	def Generate_CMDP(self, c):
		assert(len(c) == self.nC)
		assert(abs(np.sum(c) - 1) < 1e-10)
		self.Trans_prob = np.zeros((self.nS, self.nA, self.nS))
		self.Rew = np.zeros((self.nS, self.nA))
		for i, c_i in enumerate(c):
			self.Trans_prob += c_i * self.T[i]
			self.Rew += c_i * self.R[i]

	def Get_Trans_Prob_Rew(self):
		return self.Trans_prob, self.Rew

	def reset(self):
		self.t = 0
		self.s = 0
		return self.s

	def step(self, a):
		self.t += 1
		rew = self.Rew[self.s, a]
		next_s = Choose_One(self.Trans_prob[self.s, a])
		self.s = next_s
		return next_s, rew, self.t == self.H, {}