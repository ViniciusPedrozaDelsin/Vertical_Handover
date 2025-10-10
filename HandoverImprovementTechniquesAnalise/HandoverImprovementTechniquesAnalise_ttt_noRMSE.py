import matplotlib.pyplot as plt

algorithms = ['SAW', 'WPM', 'TOPSIS', 'FUZZY', 'AVERAGE']

TTT_values = {
    'SAW': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'WPM': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'TOPSIS': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'FUZZY': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'AVERAGE': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

saw_rmse_values = [1, 1.005561084, 1.007555618, 1.009991577, 1.01024956, 1.01224283, 1.011078069, 1.014495436, 1.010519097, 1.01356838, 1.013606058]
wpm_rmse_values = [1, 1.004784022, 1.007725054, 1.009732536, 1.008766218, 1.009269556, 1.01234394, 1.011783035, 1.01002068, 1.012379037, 1.013444537]
topsis_rmse_values = [1, 1.006610401, 1.008852529, 1.011674287, 1.010573059, 1.012595551, 1.012720393, 1.01544781, 1.012540014, 1.014908097, 1.014795141]
fuzzy_rmse_values = [1, 1.00617854, 1.008816677, 1.011262884, 1.011926322, 1.013588343, 1.01601411, 1.017361439, 1.014903359, 1.016213861, 1.018152297]
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

saw_vh_reduction = [x * 100 for x in [1, 0.3756097561, 0.2083333333, 0.1435897436, 0.101010101, 0.09743589744, 0.08212560386, 0.08212560386, 0.06403940887, 0.06486486486, 0.05]]
wpm_vh_reduction = [x * 100 for x in [1, 0.4060913706, 0.2445652174, 0.1534653465, 0.1105263158, 0.109375, 0.09793814433, 0.09473684211, 0.0725388601, 0.06779661017, 0.05855855856]]
topsis_vh_reduction = [x * 100 for x in [1, 0.3789954338, 0.2242990654, 0.1415525114, 0.09051724138, 0.09661835749, 0.08372093023, 0.07053941909, 0.06103286385, 0.05853658537, 0.05579399142]]
fuzzy_vh_reduction = [x * 100 for x in [1, 0.337164751, 0.1882352941, 0.1178707224, 0.08786610879, 0.07722007722, 0.06315789474, 0.06273062731, 0.04905660377, 0.046875, 0.04905660377]]
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

# First Graph: Total Handover
for algo in algorithms:
    axes[0].plot(TTT_values[algo], handover_reduction[algo], marker='o', label=algo)
axes[0].set_title('Total Handover vs Time-to-Trigger')
axes[0].set_xlabel('Time-to-Trigger Value (s)')
axes[0].set_ylabel('Total Handover (%)')
axes[0].grid(True)
axes[0].legend()

# Second Graph: RMSE Increase
for algo in algorithms:
    axes[1].plot(TTT_values[algo], rmse_increase[algo], marker='s', label=algo)
axes[1].set_title('RMSE Increase vs Time-to-Trigger')
axes[1].set_xlabel('Time-to-Trigger Value (s)')
axes[1].set_ylabel('RMSE Increase')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
