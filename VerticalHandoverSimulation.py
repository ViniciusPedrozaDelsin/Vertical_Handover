import tkinter as tk
from tkinter import ttk
import sys
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from vhSimulator import Device
from vhSimulator import WirelessNetworkSystem as WNS
from vhSimulator import SPMO_Max_Min_Method as SPMO_MMM
from vhSimulator import SPMO_Preference as SPMO_Pref
from vhSimulator import MPMO_SAW, MPMO_WPM, MPMO_TOPSIS, BenchmarkMethod, PerformanceAnalysis


# ==================================== Initial Parameters ====================================
# Map dimension
x_max, y_max = 1000, 1000

# Start position
x, y = x_max/2, y_max/2

# Interval between iterations
iter_interval = 10

# Distance for iteration
dist_iter = 10

# n = Number of iterations, j = DO NOT CHANGE
j = 0
n = 100

# Activate Graphical Interface
GUI = False

# Activate Prints for DEBBUG
verbose = False

# Number of simulations
n_simulations = 2

# Performance Analysis
analyzed_parameters = ['RSSI', 'SNR', 'Throughput', 'PC', 'MC', 'BER', 'FEC']

# Results
final_results = []
# ============================================================================================


# Wireless Network Systems
WNS_list = []

def generate_random_WNS():
    global WNS_list
    WNS_list = []
    
    # WiFi's
    global wifi_1
    wifi_1 = WNS("WiFi-1", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_1, 'green', 0.3])
    
    global wifi_2
    wifi_2 = WNS("WiFi-2", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_2, 'green', 0.3])
    
    global wifi_3
    wifi_3 = WNS("WiFi-3", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_3, 'green', 0.3])
    
    global wifi_4
    wifi_4 = WNS("WiFi-4", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_4, 'green', 0.3])
    
    global wifi_5
    wifi_5 = WNS("WiFi-5", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_5, 'green', 0.3])
    
    global wifi_6
    wifi_6 = WNS("WiFi-6", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_6, 'green', 0.3])
    
    global wifi_7
    wifi_7 = WNS("WiFi-7", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_7, 'green', 0.3])

    global wifi_8
    wifi_8 = WNS("WiFi-8", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_8, 'green', 0.3])
    
    global wifi_9
    wifi_9 = WNS("WiFi-9", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_9, 'green', 0.3])
    
    global wifi_10
    wifi_10 = WNS("WiFi-10", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_10, 'green', 0.3])


    # NB-IoT 5G
    global nbiot_5g_1
    nbiot_5g_1 = WNS("NBIoT-5g-1", random.uniform(-10*x_max, 10*x_max), random.uniform(-10*y_max, 10*y_max), 30, 800000000, 1400000, 2, "NB-IoT-5G", 0.25, 5, maximum_radius=15000, predef_throughput=[100000, 10000], predef_snr=[10, 2], predef_rssi=[-90, -115], predef_ber=[0.00001, 0.001], predef_fec=[2/3, 1/3])
    WNS_list.append([nbiot_5g_1, 'blue', 0.03])


    # LoRa's
    global LoRa_1
    LoRa_1 = WNS("LoRa-1", random.uniform(-7*x_max, 7*x_max), random.uniform(-7*y_max, 7*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 1000], predef_snr=[10, 0], predef_rssi=[-80, -120], predef_ber=[0.00001, 0.01], predef_fec=[4/5, 4/8])
    WNS_list.append([LoRa_1, 'yellow', 0.03])
    
    global LoRa_2
    LoRa_2 = WNS("LoRa-2", random.uniform(-7*x_max, 7*x_max), random.uniform(-7*y_max, 7*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 1000], predef_snr=[10, 0], predef_rssi=[-80, -120], predef_ber=[0.00001, 0.01], predef_fec=[4/5, 4/8])
    WNS_list.append([LoRa_2, 'yellow', 0.03])


    # LTE 4G
    global LTE_4g
    LTE_4g = WNS("LTE-4g-1", random.uniform(-20*x_max, 20*x_max), random.uniform(-20*y_max, 20*y_max), 40, 868000000, 250000, 5, "LTE-4G", 1.05, 3, maximum_radius=30000, predef_throughput=[100000000, 5000000], predef_snr=[15, 5], predef_rssi=[-70, -100], predef_ber=[0.000001, 0.0001], predef_fec=[3/4, 1/3])
    WNS_list.append([LTE_4g, 'red', 0.03])


    # WiFi Max
    global wifi_max_1
    wifi_max_1 = WNS("WiFi-Max-1", random.uniform(-4*x_max, 4*x_max), random.uniform(-4*y_max, 4*y_max), 40, 3000000000, 10000000, 10, "WiFi-Max", 0.80, 2, maximum_radius=6000, predef_throughput=[40000000, 2000000], predef_snr=[15, 5], predef_rssi=[-60, -90], predef_ber=[0.0000001, 0.00001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_max_1, 'purple', 0.03])


def connect_to_net(device):
    device.connect_to_network(wifi_1)
    device.connect_to_network(wifi_2)
    device.connect_to_network(wifi_3)
    device.connect_to_network(wifi_4)
    device.connect_to_network(wifi_5)
    device.connect_to_network(wifi_6)
    device.connect_to_network(wifi_7)
    device.connect_to_network(nbiot_5g_1)
    device.connect_to_network(LoRa_1)
    device.connect_to_network(LoRa_2)
    device.connect_to_network(LTE_4g)
    device.connect_to_network(wifi_max_1)



# =============================== Initialize Graph ===============================

def random_direction():
    global dist_iter
    # Random angle in radians
    angle = random.uniform(0, 2 * np.pi)
    dx = dist_iter * np.cos(angle)
    dy = dist_iter * np.sin(angle)
    return dx, dy


def update_position(device):
    global x, y, j, n, iter_interval, WNS_list, n_simulations, x_max, y_max
    dx, dy = random_direction()
    x = min(max(x + dx, 0), x_max)
    y = min(max(y + dy, 0), y_max)
    
    calculate_parameters(device, x, y)
    
    if GUI == True: plot_graph()
    
    j += 1
    if j < n:
        # Call again after 1 second
        root.after(iter_interval, lambda: update_position(device))
    else:
        performe_analysis()
        n_simulations = n_simulations - 1
        if n_simulations != 0:
            x = x_max/2
            y = y_max/2
            device = Device(1, x, y)
            connect_to_net(device)
            generate_random_WNS()
            # Cleaning old QoS parameters Storaged
            p_spmo_max_min_rssi.clean_storaged_QoS()
            p_spmo_max_min_snr.clean_storaged_QoS()
            p_spmo_pref.clean_storaged_QoS()
            p_mpmo_saw.clean_storaged_QoS()
            p_mpmo_wpm.clean_storaged_QoS()
            p_mpmo_topsis.clean_storaged_QoS()
            p_benchmark.clean_storaged_QoS()
            j = 0
            update_position(device)
        else:
            plot_results()
            sys.exit()


def calculate_parameters(device, x_position, y_position):
    if verbose == True: print(f"x:{round(x_position, 4)} || y:{round(y_position, 4)}")
    device.updatePosition(x_position, y_position)
    
    
    if verbose == True: print("===================================================")
    #print(f"Networks: {device.get_all_QoS_Parameters_predef()}")
    device.get_all_QoS_Parameters_predef()
    available_networks = device.get_available_networks()
    if verbose == True: print(f"Available Networks: {available_networks}")
    
    decision_spmo_mmm_rssi = device.makeDecision(spmo_max_min_method_rssi, available_networks)
    p_spmo_max_min_rssi.store_QoS_parameters(decision_spmo_mmm_rssi)
    if verbose == True: print(f"Decision SPMO MAX MIN RSSI: {decision_spmo_mmm_rssi}")
    
    decision_spmo_mmm_snr = device.makeDecision(spmo_max_min_method_snr, available_networks)
    p_spmo_max_min_snr.store_QoS_parameters(decision_spmo_mmm_snr)
    if verbose == True: print(f"Decision SPMO MAX MIN SNR: {decision_spmo_mmm_snr}")
    
    decision_spmo_pref = device.makeDecision(spmo_pref, available_networks)
    p_spmo_pref.store_QoS_parameters(decision_spmo_pref)
    if verbose == True: print(f"Decision SPMO Preference: {decision_spmo_pref}")
    
    decision_mpmo_saw = device.makeDecision(mpmo_saw, available_networks)
    p_mpmo_saw.store_QoS_parameters(decision_mpmo_saw)
    if verbose == True: print(f"Decision MPMO SAW: {decision_mpmo_saw}")
    
    decision_mpmo_wpm = device.makeDecision(mpmo_wpm, available_networks)
    p_mpmo_wpm.store_QoS_parameters(decision_mpmo_wpm)
    if verbose == True: print(f"Decision MPMO WPM: {decision_mpmo_wpm}")
    
    decision_mpmo_topsis = device.makeDecision(mpmo_topsis, available_networks)
    p_mpmo_topsis.store_QoS_parameters(decision_mpmo_topsis)
    if verbose == True: print(f"Decision MPMO TOPSIS: {decision_mpmo_topsis}")
    
    decision_benchmark = device.makeDecision(benchmark, available_networks)
    p_benchmark.store_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Benchmark {decision_benchmark}")
    
    if verbose == True: print("===================================================")


def performe_analysis():
    global final_results, analyzed_parameters
    
    # Gathering together the results
    results_list = []
    
    results_spmo_max_min_rssi = p_spmo_max_min_rssi.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_max_min_rssi['Handover'] = p_spmo_max_min_rssi.count_number_of_handovers()
    results_spmo_max_min_rssi['Algorithm'] = p_spmo_max_min_rssi.algorithm
    results_list.append(results_spmo_max_min_rssi)
    
    results_spmo_max_min_snr = p_spmo_max_min_snr.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_max_min_snr['Handover'] = p_spmo_max_min_snr.count_number_of_handovers()
    results_spmo_max_min_snr['Algorithm'] = p_spmo_max_min_snr.algorithm
    results_list.append(results_spmo_max_min_snr)
    
    results_spmo_pref = p_spmo_pref.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_pref['Handover'] = p_spmo_pref.count_number_of_handovers()
    results_spmo_pref['Algorithm'] = p_spmo_pref.algorithm
    results_list.append(results_spmo_pref)
    
    results_mpmo_saw = p_mpmo_saw.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_saw['Handover'] = p_mpmo_saw.count_number_of_handovers()
    results_mpmo_saw['Algorithm'] = p_mpmo_saw.algorithm
    results_list.append(results_mpmo_saw)
    
    results_mpmo_wpm = p_mpmo_wpm.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_wpm['Handover'] = p_mpmo_wpm.count_number_of_handovers()
    results_mpmo_wpm['Algorithm'] = p_mpmo_wpm.algorithm
    results_list.append(results_mpmo_wpm)
    
    results_mpmo_topsis = p_mpmo_topsis.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_topsis['Handover'] = p_mpmo_topsis.count_number_of_handovers()
    results_mpmo_topsis['Algorithm'] = p_mpmo_topsis.algorithm
    results_list.append(results_mpmo_topsis)
    
    results_benchmark = p_benchmark.calculate_average_QoS_parameters(analyzed_parameters)
    results_benchmark['Handover'] = p_benchmark.count_number_of_handovers()
    results_benchmark['Algorithm'] = p_benchmark.algorithm
    results_list.append(results_benchmark)
    
    # Print the Results
    print(f"{p_spmo_max_min_rssi.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_spmo_max_min_rssi.count_number_of_handovers()}")
    print(f"{p_spmo_max_min_snr.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_spmo_max_min_snr.count_number_of_handovers()}")
    print(f"{p_spmo_pref.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_spmo_pref.count_number_of_handovers()}")
    print(f"{p_mpmo_saw.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_mpmo_saw.count_number_of_handovers()}")
    print(f"{p_mpmo_wpm.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_mpmo_wpm.count_number_of_handovers()}")
    print(f"{p_mpmo_topsis.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_mpmo_topsis.count_number_of_handovers()}")
    print(f"{p_benchmark.calculate_average_QoS_parameters(analyzed_parameters)}, Handoff: {p_benchmark.count_number_of_handovers()}")
    print("================================================================================================================================================")
    
    final_results.append(results_list)
    


def plot_results():
    global final_results, analyzed_parameters
    analyzed_parameters.append("Handover")
    
    # Initialize aggregation storage
    aggregated_data = {}

    # Process each time step
    for time_step in final_results:
        for entry in time_step:
            algo = entry['Algorithm']
            if algo not in aggregated_data:
                aggregated_data[algo] = {param: 0 for param in analyzed_parameters}
                aggregated_data[algo]['count'] = 0
            
            for param in analyzed_parameters:
                aggregated_data[algo][param] += entry[param]
            
            aggregated_data[algo]['count'] += 1

    # Compute averages
    results = []
    for algo, values in aggregated_data.items():
        count = values.pop('count')  # Remove count after use
        results.append({param: values[param] / count for param in analyzed_parameters})
        results[-1]['Algorithm'] = algo  # Add algorithm name
    
    print(f"Results: {results}")
    
    # Organizing results by algorithm
    r_dict = {r['Algorithm']: r for r in results}

    num_params = len(analyzed_parameters)
    num_cols = 2  # Split into two vertical sections
    num_rows = math.ceil(num_params / num_cols)  # Calculate needed rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 4 * num_rows), constrained_layout=True)

    # Flatten axes for easy iteration when we have multiple rows
    axes = axes.flatten() if num_params > 1 else [axes]

    for i, param in enumerate(analyzed_parameters):
        values = [d[param] for d in r_dict.values()]
        labels = list(r_dict.keys())

        ax = axes[i]  # Get the correct subplot

        ax.bar(labels, values, color=["silver", "gold", "blue", "green", "yellow", "red", "black"], edgecolor='black', linewidth=1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylabel(param, fontsize=10)
        #ax.set_title(f"Optimization Methods X {param}", fontsize=12)

        # Set y-axis limits dynamically
        if max(values) < 0:
            ax.set_ylim(0, min(values) + min(values) * 0.1)
        else:
            ax.set_ylim(0, max(values) + max(values) * 0.1)

        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
    plt.show()


def plot_graph():
    ax.clear()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    for WirelessNetwork in WNS_list:
        circle = Circle((WirelessNetwork[0].x_position, WirelessNetwork[0].y_position), WirelessNetwork[0].maximum_radius, color=WirelessNetwork[1], alpha=WirelessNetwork[2], fill=True)
        ax.add_patch(circle)
        circle = Circle((WirelessNetwork[0].x_position, WirelessNetwork[0].y_position), 2*WirelessNetwork[0].maximum_radius/3, color=WirelessNetwork[1], alpha=WirelessNetwork[2], fill=True)
        ax.add_patch(circle)
        circle = Circle((WirelessNetwork[0].x_position, WirelessNetwork[0].y_position), WirelessNetwork[0].maximum_radius/3, color=WirelessNetwork[1], alpha=WirelessNetwork[2], fill=True)
        ax.add_patch(circle)
    
    ax.scatter(x, y, color='red', s=50)
    ax.set_title("Random Walk Simulation")
    canvas.draw()


# Create GUI
root = tk.Tk()
root.title("Random Walk Visualization")

def start_GUI():
    global ax, fig, canvas, frame
    if GUI == True:
        frame = ttk.Frame(root)
        frame.pack()

        fig, ax = plt.subplots(figsize=(8, 8))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack()

start_GUI()

device_1 = Device(1, x, y)

# Generate Random WNS
generate_random_WNS()

# Wireless Networks
connect_to_net(device_1)

# Optimization Methods
spmo_max_min_method_rssi = SPMO_MMM("SPMO-MAX-RSSI", "RSSI", True)
spmo_max_min_method_snr = SPMO_MMM("SPMO-MAX-SNR", "SNR", True)
spmo_pref = SPMO_Pref("SPMO-Preference", "Protocol", ['WiFi-5GHz', 'WiFi-2.4GHz', 'WiFi-Max', 'LTE-4G', 'NB-IoT-5G', 'LoRa-868'])
mpmo_saw = MPMO_SAW("MPMO-SAW", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
mpmo_wpm = MPMO_WPM("MPMO-WPM", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
mpmo_topsis = MPMO_TOPSIS("MPMO-TOPSIS", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
benchmark = BenchmarkMethod("Benchmark", ["RSSI", "SNR", "Throughput", "Distance", 'PC', 'MC', 'BER', 'FEC'], [1, 1, 1, 0, 0, 0, 0, 1])

# Instances of Performance Analysis
p_spmo_max_min_rssi = PerformanceAnalysis("SPMO-MAX-RSSI")
p_spmo_max_min_snr = PerformanceAnalysis("SPMO-MAX-SNR")
p_spmo_pref = PerformanceAnalysis("SPMO-Preference")
p_mpmo_saw = PerformanceAnalysis("MPMO-SAW")
p_mpmo_wpm = PerformanceAnalysis("MPMO-WPM")
p_mpmo_topsis = PerformanceAnalysis("MPMO_TOPSIS")
p_benchmark = PerformanceAnalysis("Benchmark")

update_position(device_1)
root.mainloop()
