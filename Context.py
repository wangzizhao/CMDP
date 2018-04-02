import numpy as np

def generate_uniform_context(nC):
	c = np.random.rand(nC)/nC
	c[-1] = 1 - np.sum(c[:-1])
	return c

# def generate_uniform_context(nC):
# 	c = np.random.rand(nC)
# 	c /= np.sum(c)
# 	return c

# def generate_uniform_context_with_redundant_dim(nC):
# 	"""
# 	nAdv = 2, c = (a, b, 1 - a - b) -> (a/2, a/2, b/3, b/3, b/3, 1 - a - b)
# 	"""
# 	c = np.zeros(nC)
# 	rand = np.random.rand(2)/self.nAdv
# 	c[0] = rand[0]/2
# 	c[1] = rand[0]/2
# 	c[2] = rand[1]/3
# 	c[3] = rand[1]/3
# 	c[4] = rand[1]/3
# 	c[5] = 1 - np.sum(rand)
# 	return c

# def generate_binodal_context(epi):
# 	"""
# 	Assume nAdv = 2
# 	First half of all episodes: c_t = (rand/nAdv, 0, 1 - sum(c_t[0:2]))
# 	Second half of all episodes: c_t = (rand/nAdv, rand/nAdv, 1 - sum(c_t[0:2]))
# 	"""
# 	assert(self.nAdv == 2)
# 	c = np.random.rand(nC)/self.nAdv
# 	if 2 * epi < self.episode:
# 		c[1] = c[1]/2
# 	else:
# 		c[1] = c[1]/2 + 0.5/self.nAdv
# 	c[-1] = 1 - np.sum(c[:-1])
# 	return c
