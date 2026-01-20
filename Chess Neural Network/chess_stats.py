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


#%% Student's t-test for two proportions (pooled)

p1 = (12 + 28 * 0.5) / 78  # NN wins + half draws
p2 = (38 + 28 * 0.5) / 78  # Symbolic AI wins + half draws
print(f"Proportion 1 (NN): {p1:.4f}\nProportion 2 (Symbolic AI): {p2:.4f}")

p = (26 + 52) / (78 + 78)
print(f"pooled p: {p}")

z_obs = (p2-p1) / np.sqrt(p * (1 - p) * (1/n + 1/n))
print(f"z_obs: {z_obs}")

alpha = 0.05
critical_value = stats.norm.ppf(1-alpha/2)
print(f"Critical-interval: [{-critical_value:.4f}, {critical_value:.4f}]")

p_value = 2 * (1 - stats.norm.cdf(z_obs))
print(f"P-value: {p_value:.4f}")


#%% Confidence interval of the difference in proportions

p1 = (12 + 28 * 0.5) / 1848  # NN wins + half draws
p2 = (38 + 28 * 0.5) / 1848  # Symbolic AI wins + half draws
print(f"Proportion 1 (NN): {p1:.4f}\nProportion 2 (Symbolic AI): {p2:.4f}")

alpha = 0.05
diff = p2 - p1
lower = diff - stats.norm.ppf(1-alpha/2) * np.sqrt((p1 * (1 - p1) / 78) + (p2 * (1 - p2) / 78))
upper = diff + stats.norm.ppf(1-alpha/2) * np.sqrt((p1 * (1 - p1) / 78) + (p2 * (1 - p2) / 78))

print(f"Confidence interval of the difference: [{lower:.4f}, {upper:.4f}]")

#%% NEW FORMULA
alpha = 0.05
margin_of_error = 0.015

sample_size1 = p1 * (1 - p1) * (stats.norm.ppf(1 - alpha/2)/margin_of_error)**2
sample_size2 = p2 * (1 - p2) * (stats.norm.ppf(1- alpha/2)/margin_of_error)**2
print(sample_size1, sample_size2)

nn_points = 1317.5/3796
sym_points = 2478.5/3796

difference = sym_points - nn_points

lower_nn = nn_points - 1.96 * (nn_points * (1 - nn_points)/3796) ** 0.5
upper_nn = nn_points + 1.96 * (nn_points * (1 - nn_points)/3796) ** 0.5

print(lower_nn, upper_nn)

lower_sym = sym_points - 1.96 * (sym_points * (1 - sym_points)/3796) ** 0.5
upper_sym = sym_points + 1.96 * (sym_points * (1 - sym_points)/3796) ** 0.5

print(lower_sym, upper_sym)

lower_diff = difference - 1.96 * (difference * (1 - difference)/3796) ** 0.5
upper_diff = difference + 1.96 * (difference * (1 - difference)/3796) ** 0.5

print(lower_diff, upper_diff)


#%% sample size calculation for one proportion for specific power for hyothesis testing

alpha = 0.01  # significance level
beta = 0.01   # Type II error rate
p0 = 0.5      # null hypothesis proportion
p1 = 0.45    # alternative hypothesis proportion
# effect size = p1 - p0 = 0.05
n = (stats.norm.ppf(1-alpha/2) * np.sqrt(p0 * (1 - p0)) + stats.norm.ppf(1 - beta) * np.sqrt(p1 * (1 - p1)))**2 / (p1 - p0)**2
print(f"Sample size for specific power: {np.ceil(n)}")
# %%
