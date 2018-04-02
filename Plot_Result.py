import numpy as np 
import matplotlib.pyplot as plt 
import pickle

from KIWK_LR_Rmax import *



def Running_Average(data):
	"""
	data: list or numpy array
	average_scope: int 
	"""
	averaged_data = []
	average_scope = 500
	running_sum = np.sum(data[:average_scope])
	for i in range(len(data) - average_scope):
		averaged_data.append(running_sum/average_scope)
		running_sum += data[i+average_scope] - data[i]
	averaged_data.append(running_sum/average_scope)
	return averaged_data


def Read_Data(filename):
	if filename is None:
		return None
	print("loading data...")
	with (open(filename, "rb")) as f:
		data = pickle.load(f)
	print("len of data:", len(data))
	return data


def Plot_Trans_Prob_Diff(data):
	trans_prob_diff_sum = data[5]
	plt.plot(range(len(trans_prob_diff_sum)), trans_prob_diff_sum)
	plt.title("trans_prob_diff_sum")
	plt.show()

def Plot_Rew_Diff(data):
	rew_histroy = data[6]
	rew_diff_sum = [i[0] for i in rew_histroy]
	rew_diff_sum_ra = Running_Average(rew_diff_sum)

	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(rew_diff_sum)), rew_diff_sum)
	ax[0].set_title("rew_diff_sum")
	ax[1].plot(range(len(rew_diff_sum_ra)), rew_diff_sum_ra)
	ax[1].set_title("rew_diff_sum Running Average")
	plt.show()

def Plot_Policy_Diff(data):
	policy_diff = data[7]
	policy_diff_ra = Running_Average(policy_diff)

	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(policy_diff)), policy_diff)
	ax[0].set_title("policy_diff")
	ax[1].plot(range(len(policy_diff_ra)), policy_diff_ra)
	ax[1].set_title("policy_diff Running Average")
	plt.show()


def Plot_Known_States_Num(data):
	# alpha_s_prob = data[0]
	# alpha_s_rew = data[1]
	# c_t_history = data[2]
	Q_probs = [LR_history[0] for LR_history in data[3]]
	# Q_rews = [LR_history[2] for LR_history in data[3]]
	nS = Q_probs[0].shape[0]
	nA = Q_probs[0].shape[1]

	is_state_action_pair_known = [LR_history[4] for LR_history in data[3]]
	num_known_state_action_pair = [np.sum(is_known) for is_known in is_state_action_pair_known]
	num_known_state_action_pair_ra = Running_Average(num_known_state_action_pair)

	print(nS, nA)

	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(num_known_state_action_pair)), num_known_state_action_pair)
	ax[0].set_title("num_known_state_action_pair")
	ax[1].plot(range(len(num_known_state_action_pair_ra)), num_known_state_action_pair_ra)
	ax[1].set_title("num_known_state_action_pair Running Average")
	plt.show()

def Plot_Wrong_Action_Num(data):
	# print("episode_history len", len(data[4]))
	# num_total_wrong_action = [episode_history[2] for episode_history in data[4]]
	num_wrong_action = [episode_history[1] for episode_history in data]
	num_wrong_action_ra = Running_Average(num_wrong_action)
	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(num_wrong_action)), num_wrong_action)
	ax[0].set_title("num_wrong_action")
	ax[0].set_xlabel("episode")
	ax[1].plot(range(len(num_wrong_action_ra)), num_wrong_action_ra)
	ax[1].set_title("num_wrong_action Running Average")
	ax[1].set_xlabel("episode")
	plt.show()

def Plot_Eps_Rew_Diff(data):
	eps_rew_diff = [episode_history[2] for episode_history in data]
	eps_rew_diff_ra = Running_Average(eps_rew_diff)
	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(eps_rew_diff)), eps_rew_diff)
	ax[0].set_title("eps_rew_diff")
	ax[0].set_xlabel("episode")
	ax[1].plot(range(len(eps_rew_diff_ra)), eps_rew_diff_ra)
	ax[1].set_title("eps_rew_diff Running Average")
	ax[1].set_xlabel("episode")
	plt.show()

def Plot_Value_Diff(data):
	init_state_value_diff = [episode_history[0] for episode_history in data]
	init_state_value_diff_ra = Running_Average(init_state_value_diff)
	f, ax = plt.subplots(1, 2)
	ax[0].plot(range(len(init_state_value_diff)), init_state_value_diff)
	ax[0].set_title("init_state_value_diff")
	ax[0].set_xlabel("episode")
	ax[1].plot(range(len(init_state_value_diff_ra)), init_state_value_diff_ra)
	ax[1].set_title("init_state_value_diff Running Average")
	ax[1].set_xlabel("episode")
	plt.show()


if __name__ == "__main__":
	filename = "data\CMDP\_nC_4_nA_4_H_4_CtxModel_0_20180329-230654" + ".p"
	data = Read_Data(filename)
	# Plot_Known_States_Num(data)
	# Plot_Trans_Prob_Diff(data)
	# Plot_Rew_Diff(data)
	# Plot_Policy_Diff(data)

	Plot_Wrong_Action_Num(data)
	Plot_Eps_Rew_Diff(data)
	Plot_Value_Diff(data)