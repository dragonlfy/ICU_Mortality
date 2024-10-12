# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
noise = np.random.normal(0, 1, 400) / 3

plt.figure(figsize=(20, 8))
plt.plot(noise)
plt.show()

# %%
