import numpy as np
from scipy.stats import entropy

def Trans_Prob_Metric(trans_prob1, trans_prob2):
	"""
	Input: 
		np.array((nS, nA, nS))
	Output:
		trans_prob_diff: KL divergence of each trans_prob(s, a) pair
		trans_prob_diff_sum: sum of trans_prob_diff
		trans_prob_diff_max: max of trans_prob_diff
	"""
	nS = trans_prob1.shape[0]
	nA = trans_prob1.shape[1]
	trans_prob_diff = np.zeros((nS, nA))
	for s in range(nS):
		for a in range(nA):
			trans_prob_diff[s, a] = np.sum(abs(trans_prob2[s, a] - trans_prob1[s, a]))
	trans_prob_diff_sum = np.sum(trans_prob_diff)
	trans_prob_diff_max = np.max(trans_prob_diff)

	return trans_prob_diff, trans_prob_diff_sum, trans_prob_diff_max

def Rew_Metric(rew1, rew2):
	"""
	Input:
		rew: np.array((nS, nA))
	"""
	rew_diff = abs(rew1 - rew2)
	rew_diff_sum = np.sum(rew_diff)
	rew_diff_max = np.max(rew_diff)
	return rew_diff, rew_diff_sum, rew_diff_max

def Policy_Metric(policy1, policy2):
	"""
	metric based on value function:
	Input:
		policy: np.array((H, nS))
	"""
	policy_diff = np.zeros_like(policy1)
	policy_diff[policy1 != policy2] = 1
	policy_diff_sum = np.sum(policy_diff)
	return policy_diff, policy_diff_sum

def Value_Metric(value1, value2):
	"""
	metric based on value function:
	Input:
		value: np.array((H, nS))
	"""
	value_diff = abs(value1 - value2)
	init_state_value_diff = value_diff[0][0]
	return init_state_value_diff
