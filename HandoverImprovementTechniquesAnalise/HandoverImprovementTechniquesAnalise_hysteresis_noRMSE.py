import matplotlib.pyplot as plt

algorithms = ['SAW', 'WPM', 'TOPSIS', 'FUZZY', 'AVERAGE']

hysteresis_values = {
    'SAW': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'WPM': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'TOPSIS': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'FUZZY': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'AVERAGE': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

saw_rmse_values = [1, 1.000903655, 1.001493858, 1.013572733, 1.02100702, 1.030440808, 1.042932799, 1.05906268, 1.075295881, 1.078541689, 1.116444137]
wpm_rmse_values = [1, 0.9993097966, 1.000539815, 1.001984227, 1.005430751, 1.007903668, 1.005224549, 1.008109877, 1.011269684, 1.016595292, 1.012203366]
topsis_rmse_values = [1, 1.002751154, 1.000315037, 1.008186614, 1.019218266, 1.052851661, 1.074618404, 1.09854885, 1.148252531, 1.163597531, 1.199511881]
fuzzy_rmse_values = [1, 0.9985331117, 0.9968852346, 0.9977360982, 1.003781216, 1.004419677, 1.00984106, 1.008176323, 1.013320703, 1.023192436, 1.020516514]
average_rmse_values = []
for i in range(11):
    average_rmse_values.append((saw_rmse_values[i]+wpm_rmse_values[i]+topsis_rmse_values[i]+fuzzy_rmse_values[i])/4)

rmse_increase = {
    'SAW': saw_rmse_values,
    'WPM': wpm_rmse_values,
    'TOPSIS': topsis_rmse_values,
    'FUZZY': fuzzy_rmse_values,
    'AVERAGE': average_rmse_values
}

saw_vh_reduction = [1, 0.5756097561, 0.4675925926, 0.4153846154, 0.3282828283, 0.3179487179, 0.2125603865, 0.1642512077, 0.09359605911, 0.08108108108, 0.0375]
wpm_vh_reduction = [1, 0.6802030457, 0.6413043478, 0.5841584158, 0.5263157895, 0.5572916667, 0.4639175258, 0.4526315789, 0.3626943005, 0.3785310734, 0.2747747748]
topsis_vh_reduction = [1, 0.5433789954, 0.4392523364, 0.3470319635, 0.1982758621, 0.1449275362, 0.06511627907, 0.04979253112, 0.0234741784, 0.03414634146, 0.03004291845]
fuzzy_vh_reduction = [1, 0.5517241379, 0.4470588235, 0.3992395437, 0.3807531381, 0.3204633205, 0.2631578947, 0.258302583, 0.2377358491, 0.203125, 0.1849056604]
average_vh_reduction = []
for i in range(11):
    average_vh_reduction.append((saw_vh_reduction[i]+wpm_vh_reduction[i]+topsis_vh_reduction[i]+fuzzy_vh_reduction[i])/4)

handover_reduction = {
    'SAW': saw_vh_reduction,
    'WPM': wpm_vh_reduction,
    'TOPSIS': topsis_vh_reduction,
    'FUZZY': fuzzy_vh_reduction,
    'AVERAGE': average_vh_reduction
}
    

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

# First Graph: Handover Reduction
for algo in algorithms:
    axes[0].plot(hysteresis_values[algo], handover_reduction[algo], marker='o', label=algo)
axes[0].set_title('Handover Reduction vs Hysteresis Margin')
axes[0].set_xlabel('Hysteresis Margin (%)')
axes[0].set_ylabel('Handover Reduction')
axes[0].grid(True)
axes[0].legend()

# Second Graph: RMSE Increase
for algo in algorithms:
    axes[1].plot(hysteresis_values[algo], rmse_increase[algo], marker='s', label=algo)
axes[1].set_title('RMSE Increase vs Hysteresis Margin')
axes[1].set_xlabel('Hysteresis Margin (%)')
axes[1].set_ylabel('RMSE Increase')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
