import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

def moore_law(t, T0, tau):
    return T0 * 2**((t - years[0]) / tau)

data = pd.read_csv('transistor_counts.csv', header=None)
years = data.iloc[:, 0].values
transistor_counts = data.iloc[:, 1].values

initial_guess = [transistor_counts[0], 2]

params, covariance = curve_fit(moore_law, years, transistor_counts, p0=initial_guess)

T0_fit, tau_fit = params

predicted_counts = moore_law(years, T0_fit, tau_fit)

plt.figure(figsize=(10, 6))
plt.scatter(years, transistor_counts, label='Original Data')
plt.plot(years, predicted_counts, color='red', label='Fitted Model')
plt.xlabel('Year')
plt.ylabel('Transistor Counts')
plt.title("Moore's Law Fitting")
plt.legend()
plt.grid(True)
plt.savefig('moore_law_fit.png')
plt.show()

print("Fitted Parameters:")
print("T0 =", T0_fit)
print("tau =", tau_fit)
