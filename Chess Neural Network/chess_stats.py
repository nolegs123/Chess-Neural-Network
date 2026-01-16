#%% Library imports
import numpy as np
import scipy.stats as stats

#%% Sample size calculation for two proportions for power analysis
# Pilot study results (n=30 games):

alpha = 0.01    # chance of false positive
beta = 0.01     # chance of false negative
p1 = 9.5/30     # NN baseline/pilot study win rate
p2 = 20.5/30    # Symbolic AI baseline/pilot study win rate


n = ((stats.norm.ppf(alpha/2) + stats.norm.ppf(beta))**2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)**2

print(f"Sample size: {n}")

#%% Sample size calculation for two proportions for specific confidence interval

ME = 0.03  # margin of error
alpha = 0.05  # significance level

n = stats.norm.ppf(alpha/2)**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / ME**2
print(f"Sample size for specific ME: {n}")

#%% Confidence interval of the difference in proportions

p1 = (12 + 28 * 0.5) / 78  # NN wins + half draws
p2 = (38 + 28 * 0.5) / 78  # Symbolic AI wins + half draws

alpha = 0.05
diff = p2 - p1
lower = diff - stats.norm.ppf(1-alpha/2) * np.sqrt((p1 * (1 - p1) / 78) + (p2 * (1 - p2) / 78))
upper = diff + stats.norm.ppf(1-alpha/2) * np.sqrt((p1 * (1 - p1) / 78) + (p2 * (1 - p2) / 78))

print(f"Confidence interval of the difference: [{lower:.4f}, {upper:.4f}]")