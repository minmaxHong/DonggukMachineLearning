import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("transistor_counts.csv", header=None)

years = data.iloc[:, 0].values
transistor_counts = data.iloc[:, 1].values

log_transistor_counts = np.log(transistor_counts)

coefficients = np.polyfit(years, log_transistor_counts, 1)
linear_fit = np.polyval(coefficients, years)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(years, transistor_counts, 'o-', label='Original Data')
plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Transistor Count')
plt.title('Original Data (Log Scale)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(years, log_transistor_counts, 'o-', label='Log Transformed Data')
plt.plot(years, linear_fit, 'r--', label='Linear Fit')
plt.xlabel('Year')
plt.ylabel('Log(Transistor Count)')
plt.title('Log Transformed Data with Linear Fit')
plt.legend()

plt.tight_layout()

plt.savefig('moore_law_plot.png')
plt.show()
