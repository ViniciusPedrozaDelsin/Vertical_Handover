import matplotlib.pyplot as plt

algorithms = ['SAW', 'WPM', 'TOPSIS', 'FUZZY', 'AVERAGE']

lockin_values = {
    'SAW': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'WPM': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TOPSIS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'FUZZY': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'AVERAGE': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

saw_rmse_values = [1, 1.10052172, 1.092077256, 1.1306527, 1.226695017, 1.235956732, 1.213771797, 1.184264494, 1.230570694, 1.206282469, 1.22225261]
wpm_rmse_values = [1, 1.099176334, 1.092417051, 1.141563527, 1.219013354, 1.224797599, 1.201457568, 1.191153854, 1.241387236, 1.216679675, 1.218388673]
topsis_rmse_values = [1, 1.111458758, 1.093739201, 1.155202791, 1.243431046, 1.233626129, 1.220201806, 1.20703813, 1.260392478, 1.241945194, 1.251821309]
fuzzy_rmse_values = [1, 1.130619206, 1.167674712, 1.183789122, 1.230398182, 1.253580392, 1.246206135, 1.235373883, 1.249092544, 1.202455499, 1.214722745]
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

saw_vh_reduction = [1, 0.4390243902, 0.3888888889, 0.2461538462, 0.09595959596, 0.06153846154, 0.07729468599, 0.1062801932, 0.02955665025, 0.02702702703, 0.02083333333]
wpm_vh_reduction = [1, 0.4517766497, 0.402173913, 0.2227722772, 0.08947368421, 0.05729166667, 0.07216494845, 0.07368421053, 0.0207253886, 0.005649717514, 0.009009009009]
topsis_vh_reduction = [1, 0.3881278539, 0.3925233645, 0.2237442922, 0.0775862069, 0.06763285024, 0.06976744186, 0.07053941909, 0.02816901408, 0.01951219512, 0.004291845494]
fuzzy_vh_reduction = [1, 0.2030651341, 0.1215686275, 0.04942965779, 0, 0, 0.00701754386, 0.007380073801, 0, 0.00390625, 0.003773584906]
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
    axes[0].plot(lockin_values[algo], handover_reduction[algo], marker='o', label=algo)
axes[0].set_title('Handover Reduction vs Threshold Margin')
axes[0].set_xlabel('Threshold Margin (%)')
axes[0].set_ylabel('Handover Reduction')
axes[0].grid(True)
axes[0].legend()

# Second Graph: RMSE Increase
for algo in algorithms:
    axes[1].plot(lockin_values[algo], rmse_increase[algo], marker='s', label=algo)
axes[1].set_title('RMSE Increase vs Threshold Margin')
axes[1].set_xlabel('Threshold Margin (%)')
axes[1].set_ylabel('RMSE Increase')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
