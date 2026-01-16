import numpy as np
import scipy.stats as stats

# Sample size calculation for two proportions for power analysis

alpha = 0.05    # chance of false positive
beta = 0.05     # chance of false negative
p1 = 7/30
p2 = 14/30


n = ((stats.norm.ppf(alpha/2) + stats.norm.ppf(beta))**2 * (p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)**2

print(f"Sample size: {n}")

# Sample size calculation for two proportions for specific confidence interval

ME = 0.03  # margin of error

n = stats.norm.ppf(alpha/2)**2 * (p1 * (1 - p1) + p2 * (1 - p2)) / ME**2
print(f"Sample size for specific ME: {n}")