import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
H = 0.1
times = np.arange(10, dtype=float)   # t0=0, ..., t9=9
n = len(times)

# -----------------------------
# fBM covariance function
# Cov(B_H(t), B_H(s)) =
# 0.5 * (t^(2H) + s^(2H) - |t-s|^(2H))
# -----------------------------
def fbm_cov(t, s, H):
    return 0.5 * (t**(2*H) + s**(2*H) - np.abs(t - s)**(2*H))

# Build covariance matrix
cov = np.empty((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        cov[i, j] = fbm_cov(times[i], times[j], H)

print("Covariance matrix:")
print(cov)

# -----------------------------
# Plot covariance matrix
# -----------------------------
plt.figure(figsize=(6, 5))
im = plt.imshow(cov, origin="lower")
plt.colorbar(im, label="Covariance")
plt.xticks(range(n), [f"t{i}" for i in range(n)])
plt.yticks(range(n), [f"t{i}" for i in range(n)])
plt.title(f"fBM covariance matrix (H={H})")
plt.tight_layout()
plt.savefig('fbm_covariance_matrix.png')

# -----------------------------
# Sample one fBM path
# Mean is zero
# -----------------------------
mean = np.zeros(n)
x = np.random.multivariate_normal(mean, cov)

print("\nSampled fBM values:")
for i, xi in enumerate(x):
    print(f"t{i}={times[i]:.0f}, B_H(t{i})={xi:.6f}")

# -----------------------------
# Plot sampled path
# -----------------------------
plt.figure(figsize=(7, 4))
plt.plot(times, x, marker="o")
plt.axhline(0, linewidth=1)
plt.xticks(times)
plt.xlabel("t")
plt.ylabel("B_H(t)")
plt.title(f"Sampled fractional Brownian motion path (H={H})")
plt.tight_layout()
plt.savefig("fbm_sample.png")