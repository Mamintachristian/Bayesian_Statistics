# Average Coffee Consumption of Filipinos (Cups per Day)
import numpy as np
import matplotlib.pyplot as plt

# Generating Synthetic Data Based on a Website's True Mean
np.random.seed(42)
true_mu = 2.5  # Assumed true mean from a website
true_sigma = 1  # Assumed standard deviation

data = np.random.normal(true_mu, true_sigma, size=200)

# Prior Belief: Filipinos drink an average of 2 cups per day with a standard deviation of 1
prior_mu_mean = 2
prior_mu_precision = 1 / 1**2 
prior_sigma_alpha = 2
prior_sigma_beta = 2 

# Updating the prior with observed data
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data) / true_sigma**2) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Sampling from the posterior distributions
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plot the posterior distributions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior distribution of Filipino Coffee Consumption ($\mu$)')
plt.xlabel('Average cups per day ($\mu$)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior distribution of $\sigma$')
plt.xlabel('$\sigma$')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu (updated belief on coffee consumption):", mean_mu)
print("Standard deviation of mu:", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma:", mean_sigma)
print("Standard deviation of sigma:", std_sigma)
