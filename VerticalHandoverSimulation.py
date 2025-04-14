import tkinter as tk
from tkinter import ttk
import sys
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
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
iter_interval = 1

# Distance for iteration
dist_iter = 10

# n = Number of iterations, j = DO NOT CHANGE
j = 0
n = 10000

# Activate Graphical Interface
GUI = False

# Activate Prints for DEBBUG
verbose = False

# Number of simulations
n_simulations = 1000

# Performance Analysis
analyzed_parameters = ['RSSI', 'SNR', 'Throughput', 'PC', 'MC', 'BER', 'FEC']
weights = [2, 2, 2, 1, 1, 1, 1]
#weights = [2, 4, 9, 1, 1, 1, 2]
directions = [1, 1, 1, 0, 0, 0, 1]
hyst_percentage = 0.1
tt_trigger = 2

# Results
final_results = []
indicators_results = []
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
    LTE_4g = WNS("LTE-4g-1", random.uniform(-20*x_max, 20*x_max), random.uniform(-20*y_max, 20*y_max), 40, 1900000000, 20000000, 5, "LTE-4G", 1.05, 3, maximum_radius=30000, predef_throughput=[100000000, 5000000], predef_snr=[15, 5], predef_rssi=[-70, -100], predef_ber=[0.000001, 0.0001], predef_fec=[3/4, 1/3])
    WNS_list.append([LTE_4g, 'red', 0.03])


    # WiFi Max
    global wifi_max_1
    wifi_max_1 = WNS("WiFi-Max-1", random.uniform(-4*x_max, 4*x_max), random.uniform(-4*y_max, 4*y_max), 40, 3000000000, 10000000, 10, "WiFi-Max", 0.80, 2, maximum_radius=6000, predef_throughput=[40000000, 2000000], predef_snr=[15, 5], predef_rssi=[-60, -90], predef_ber=[0.0000001, 0.00001], predef_fec=[5/6, 1/2])
    WNS_list.append([wifi_max_1, 'purple', 0.03])
    
    # Print for DEBBUG
    if verbose == True: [print(wns) for wns in WNS_list]


def connect_to_net(device):
    device.connect_to_network(wifi_1)
    device.connect_to_network(wifi_2)
    device.connect_to_network(wifi_3)
    device.connect_to_network(wifi_4)
    device.connect_to_network(wifi_5)
    device.connect_to_network(wifi_6)
    device.connect_to_network(wifi_7)
    device.connect_to_network(wifi_8)
    device.connect_to_network(wifi_9)
    device.connect_to_network(wifi_10)
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
            generate_random_WNS()
            connect_to_net(device)
            # Cleaning old QoS parameters Storaged
            p_spmo_max_min_rssi.clean_storaged_QoS()
            p_spmo_max_min_snr.clean_storaged_QoS()
            p_spmo_pref.clean_storaged_QoS()
            p_mpmo_saw.clean_storaged_QoS()
            p_mpmo_saw_hyst.clean_storaged_QoS()
            p_mpmo_saw_ttt.clean_storaged_QoS()
            p_mpmo_wpm.clean_storaged_QoS()
            p_mpmo_wpm_hyst.clean_storaged_QoS()
            p_mpmo_wpm_ttt.clean_storaged_QoS()
            p_mpmo_topsis.clean_storaged_QoS()
            p_mpmo_topsis_hyst.clean_storaged_QoS()
            p_mpmo_topsis_ttt.clean_storaged_QoS()
            p_benchmark.clean_storaged_QoS()
            # Cleaning old Benchmark QoS parameters Storaged
            p_spmo_max_min_rssi.clean_Benchmark_storaged_QoS()
            p_spmo_max_min_snr.clean_Benchmark_storaged_QoS()
            p_spmo_pref.clean_Benchmark_storaged_QoS()
            p_mpmo_saw.clean_Benchmark_storaged_QoS()
            p_mpmo_saw_hyst.clean_Benchmark_storaged_QoS()
            p_mpmo_saw_ttt.clean_Benchmark_storaged_QoS()
            p_mpmo_wpm.clean_Benchmark_storaged_QoS()
            p_mpmo_wpm_hyst.clean_Benchmark_storaged_QoS()
            p_mpmo_wpm_ttt.clean_Benchmark_storaged_QoS()
            p_mpmo_topsis.clean_Benchmark_storaged_QoS()
            p_mpmo_topsis_hyst.clean_Benchmark_storaged_QoS()
            p_mpmo_topsis_ttt.clean_Benchmark_storaged_QoS()
            p_benchmark.clean_Benchmark_storaged_QoS()
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
    
    decision_benchmark = device.makeDecision(benchmark, available_networks)
    p_benchmark.store_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Benchmark {decision_benchmark}")
    
    decision_spmo_mmm_rssi = device.makeDecision(spmo_max_min_method_rssi, available_networks)
    p_spmo_max_min_rssi.store_QoS_parameters(decision_spmo_mmm_rssi)
    p_spmo_max_min_rssi.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision SPMO MAX MIN RSSI: {decision_spmo_mmm_rssi}")
    
    decision_spmo_mmm_snr = device.makeDecision(spmo_max_min_method_snr, available_networks)
    p_spmo_max_min_snr.store_QoS_parameters(decision_spmo_mmm_snr)
    p_spmo_max_min_snr.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision SPMO MAX MIN SNR: {decision_spmo_mmm_snr}")
    
    decision_spmo_pref = device.makeDecision(spmo_pref, available_networks)
    p_spmo_pref.store_QoS_parameters(decision_spmo_pref)
    p_spmo_pref.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision SPMO Preference: {decision_spmo_pref}")
    
    decision_mpmo_saw = device.makeDecision(mpmo_saw, available_networks)
    p_mpmo_saw.store_QoS_parameters(decision_mpmo_saw)
    p_mpmo_saw.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO SAW: {decision_mpmo_saw}")
    
    decision_mpmo_saw_hyst = device.makeDecision(mpmo_saw_hyst, available_networks)
    p_mpmo_saw_hyst.store_QoS_parameters(decision_mpmo_saw_hyst)
    p_mpmo_saw_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO SAW Hysteresis: {decision_mpmo_saw_hyst}")
    
    decision_mpmo_saw_ttt = device.makeDecision(mpmo_saw_ttt, available_networks)
    p_mpmo_saw_ttt.store_QoS_parameters(decision_mpmo_saw_ttt)
    p_mpmo_saw_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO SAW Time to Trigger: {decision_mpmo_saw_ttt}")
    
    decision_mpmo_wpm = device.makeDecision(mpmo_wpm, available_networks)
    p_mpmo_wpm.store_QoS_parameters(decision_mpmo_wpm)
    p_mpmo_wpm.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO WPM: {decision_mpmo_wpm}")
    
    decision_mpmo_wpm_hyst = device.makeDecision(mpmo_wpm_hyst, available_networks)
    p_mpmo_wpm_hyst.store_QoS_parameters(decision_mpmo_wpm_hyst)
    p_mpmo_wpm_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO WPM Hysteresis: {decision_mpmo_wpm_hyst}")
    
    decision_mpmo_wpm_ttt = device.makeDecision(mpmo_wpm_ttt, available_networks)
    p_mpmo_wpm_ttt.store_QoS_parameters(decision_mpmo_wpm_ttt)
    p_mpmo_wpm_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO WPM Time to Trigger: {decision_mpmo_wpm_ttt}")
    
    decision_mpmo_topsis = device.makeDecision(mpmo_topsis, available_networks)
    p_mpmo_topsis.store_QoS_parameters(decision_mpmo_topsis)
    p_mpmo_topsis.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO TOPSIS: {decision_mpmo_topsis}")
    #if verbose == True: print(f"\033[91mDecision MPMO TOPSIS: {decision_mpmo_topsis}\033[0m")
    
    decision_mpmo_topsis_hyst = device.makeDecision(mpmo_topsis_hyst, available_networks)
    p_mpmo_topsis_hyst.store_QoS_parameters(decision_mpmo_topsis_hyst)
    p_mpmo_topsis_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO TOPSIS Hysteresis: {decision_mpmo_topsis_hyst}")
    
    decision_mpmo_topsis_ttt = device.makeDecision(mpmo_topsis_ttt, available_networks)
    p_mpmo_topsis_ttt.store_QoS_parameters(decision_mpmo_topsis_ttt)
    p_mpmo_topsis_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Decision MPMO TOPSIS Time to Trigger: {decision_mpmo_topsis_ttt}")
    #if verbose == True: print(f"\033[91mDecision MPMO TOPSIS Time to Trigger: {decision_mpmo_topsis_ttt}\033[0m")
    
    if verbose == True: print("===================================================")


def performe_analysis():
    global final_results, analyzed_parameters, indicators_results
    
    # Gathering together the results
    results_list = []
    indicators_list = []
    
    results_spmo_max_min_rssi = p_spmo_max_min_rssi.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_max_min_rssi['Handover'] = p_spmo_max_min_rssi.count_number_of_handovers()
    results_spmo_max_min_rssi['Algorithm'] = p_spmo_max_min_rssi.algorithm
    indicators_spmo_max_min_rssi = p_spmo_max_min_rssi.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_spmo_max_min_rssi['Algorithm'] = results_spmo_max_min_rssi['Algorithm']
    indicators_list.append(indicators_spmo_max_min_rssi)
    results_list.append(results_spmo_max_min_rssi)
    
    results_spmo_max_min_snr = p_spmo_max_min_snr.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_max_min_snr['Handover'] = p_spmo_max_min_snr.count_number_of_handovers()
    results_spmo_max_min_snr['Algorithm'] = p_spmo_max_min_snr.algorithm
    indicators_spmo_max_min_snr = p_spmo_max_min_snr.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_spmo_max_min_snr['Algorithm'] = results_spmo_max_min_snr['Algorithm']
    indicators_list.append(indicators_spmo_max_min_snr)
    results_list.append(results_spmo_max_min_snr)
    
    results_spmo_pref = p_spmo_pref.calculate_average_QoS_parameters(analyzed_parameters)
    results_spmo_pref['Handover'] = p_spmo_pref.count_number_of_handovers()
    results_spmo_pref['Algorithm'] = p_spmo_pref.algorithm
    indicators_spmo_pref = p_spmo_pref.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_spmo_pref['Algorithm'] = results_spmo_pref['Algorithm']
    indicators_list.append(indicators_spmo_pref)
    results_list.append(results_spmo_pref)
    
    results_mpmo_saw = p_mpmo_saw.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_saw['Handover'] = p_mpmo_saw.count_number_of_handovers()
    results_mpmo_saw['Algorithm'] = p_mpmo_saw.algorithm
    indicators_mpmo_saw = p_mpmo_saw.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_saw['Algorithm'] = results_mpmo_saw['Algorithm']
    indicators_list.append(indicators_mpmo_saw)
    results_list.append(results_mpmo_saw)
    
    results_mpmo_saw_hyst = p_mpmo_saw_hyst.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_saw_hyst['Handover'] = p_mpmo_saw_hyst.count_number_of_handovers()
    results_mpmo_saw_hyst['Algorithm'] = p_mpmo_saw_hyst.algorithm
    indicators_mpmo_saw_hyst = p_mpmo_saw_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_saw_hyst['Algorithm'] = results_mpmo_saw_hyst['Algorithm']
    indicators_list.append(indicators_mpmo_saw_hyst)
    results_list.append(results_mpmo_saw_hyst)
    
    results_mpmo_saw_ttt = p_mpmo_saw_ttt.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_saw_ttt['Handover'] = p_mpmo_saw_ttt.count_number_of_handovers()
    results_mpmo_saw_ttt['Algorithm'] = p_mpmo_saw_ttt.algorithm
    indicators_mpmo_saw_ttt = p_mpmo_saw_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_saw_ttt['Algorithm'] = results_mpmo_saw_ttt['Algorithm']
    indicators_list.append(indicators_mpmo_saw_ttt)
    results_list.append(results_mpmo_saw_ttt)
    
    results_mpmo_wpm = p_mpmo_wpm.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_wpm['Handover'] = p_mpmo_wpm.count_number_of_handovers()
    results_mpmo_wpm['Algorithm'] = p_mpmo_wpm.algorithm
    indicators_mpmo_wpm = p_mpmo_wpm.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_wpm['Algorithm'] = results_mpmo_wpm['Algorithm']
    indicators_list.append(indicators_mpmo_wpm)
    results_list.append(results_mpmo_wpm)
    
    results_mpmo_wpm_hyst = p_mpmo_wpm_hyst.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_wpm_hyst['Handover'] = p_mpmo_wpm_hyst.count_number_of_handovers()
    results_mpmo_wpm_hyst['Algorithm'] = p_mpmo_wpm_hyst.algorithm
    indicators_mpmo_wpm_hyst = p_mpmo_wpm_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_wpm_hyst['Algorithm'] = results_mpmo_wpm_hyst['Algorithm']
    indicators_list.append(indicators_mpmo_wpm_hyst)
    results_list.append(results_mpmo_wpm_hyst)
    
    results_mpmo_wpm_ttt = p_mpmo_wpm_ttt.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_wpm_ttt['Handover'] = p_mpmo_wpm_ttt.count_number_of_handovers()
    results_mpmo_wpm_ttt['Algorithm'] = p_mpmo_wpm_ttt.algorithm
    indicators_mpmo_wpm_ttt = p_mpmo_wpm_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_wpm_ttt['Algorithm'] = results_mpmo_wpm_ttt['Algorithm']
    indicators_list.append(indicators_mpmo_wpm_ttt)
    results_list.append(results_mpmo_wpm_ttt)
    
    results_mpmo_topsis = p_mpmo_topsis.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_topsis['Handover'] = p_mpmo_topsis.count_number_of_handovers()
    results_mpmo_topsis['Algorithm'] = p_mpmo_topsis.algorithm
    indicators_mpmo_topsis = p_mpmo_topsis.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_topsis['Algorithm'] = results_mpmo_topsis['Algorithm']
    indicators_list.append(indicators_mpmo_topsis)
    results_list.append(results_mpmo_topsis)
    
    results_mpmo_topsis_hyst = p_mpmo_topsis_hyst.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_topsis_hyst['Handover'] = p_mpmo_topsis_hyst.count_number_of_handovers()
    results_mpmo_topsis_hyst['Algorithm'] = p_mpmo_topsis_hyst.algorithm
    indicators_mpmo_topsis_hyst = p_mpmo_topsis_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_topsis_hyst['Algorithm'] = results_mpmo_topsis_hyst['Algorithm']
    indicators_list.append(indicators_mpmo_topsis_hyst)
    results_list.append(results_mpmo_topsis_hyst)
    
    results_mpmo_topsis_ttt = p_mpmo_topsis_ttt.calculate_average_QoS_parameters(analyzed_parameters)
    results_mpmo_topsis_ttt['Handover'] = p_mpmo_topsis_ttt.count_number_of_handovers()
    results_mpmo_topsis_ttt['Algorithm'] = p_mpmo_topsis_ttt.algorithm
    indicators_mpmo_topsis_ttt = p_mpmo_topsis_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_mpmo_topsis_ttt['Algorithm'] = results_mpmo_topsis_ttt['Algorithm']
    indicators_list.append(indicators_mpmo_topsis_ttt)
    results_list.append(results_mpmo_topsis_ttt)
    
    results_benchmark = p_benchmark.calculate_average_QoS_parameters(analyzed_parameters)
    results_benchmark['Handover'] = p_benchmark.count_number_of_handovers()
    results_benchmark['Algorithm'] = p_benchmark.algorithm
    results_list.append(results_benchmark)
    
    # Print the Results
    print(f"{results_spmo_max_min_rssi}, Handoff: {results_spmo_max_min_rssi['Handover']}")
    print(f"{results_spmo_max_min_snr}, Handoff: {results_spmo_max_min_snr['Handover']}")
    print(f"{results_spmo_pref}, Handoff: {results_spmo_pref['Handover']}")
    print(f"{results_mpmo_saw}, Handoff: {results_mpmo_saw['Handover']}")
    print(f"{results_mpmo_saw_hyst}, Handoff: {results_mpmo_saw_hyst['Handover']}")
    print(f"{results_mpmo_wpm}, Handoff: {results_mpmo_wpm['Handover']}")
    print(f"{results_mpmo_wpm_hyst}, Handoff: {results_mpmo_wpm_hyst['Handover']}")
    print(f"{results_mpmo_wpm_ttt}, Handoff: {results_mpmo_wpm_ttt['Handover']}")
    print(f"{results_mpmo_topsis}, Handoff: {results_mpmo_topsis['Handover']}")
    print(f"{results_mpmo_topsis_hyst}, Handoff: {results_mpmo_topsis_hyst['Handover']}")
    print(f"{results_mpmo_topsis_ttt}, Handoff: {results_mpmo_topsis_ttt['Handover']}")
    print(f"{results_benchmark}, Handoff: {results_benchmark['Handover']}")
    print("================================================================================================================================================")
    
    final_results.append(results_list)
    indicators_results.append(indicators_list)
    


def plot_results():
    global final_results, analyzed_parameters, indicators_results
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
    
    print("================================================")
    print("Indicators to Measure Deviation from a Benchmark")

    # Initialize a dictionary to store the summed values
    aggregated_results_rmse = {}

    # Loop through each list in the indicators_results
    for group in indicators_results:
        for entry in group:
            algorithm = entry['Algorithm']
            
            if algorithm not in aggregated_results_rmse:
                aggregated_results_rmse[algorithm] = {param: 0 for param in entry if param != 'Algorithm'}
            
            for param in entry:
                if param != 'Algorithm':
                    aggregated_results_rmse[algorithm][param] += ((sum(entry[param]) / len(entry[param]))**(1/2))
    
    # Print the aggregated results
    ind_dict = {}
    for algorithm, values in aggregated_results_rmse.items():
        for param, value in values.items():
            ind_dict[param] = []
            
    for algorithm, values in aggregated_results_rmse.items():
        print(f"Algorithm: {algorithm}")
        for param, value in values.items():
            ind_dict[param].append(value)
            print(f"RMSE: {param}: {value}")
        print("-" * 50)
    
    print("RMSE of the Normalized Results")
    normalized_rsme = {
        k: [v_i / max(v) for v_i in v]
        for k, v in ind_dict.items()
    }
    print(normalized_rsme)
    
    zipped = zip(*normalized_rsme.values())
    # Calculate RMSE by Indice
    #rmse_per_index = [np.sqrt(np.mean([val**2 for val in values])) for values in zipped]
    rmse_per_index = [
        np.sqrt(np.mean([
            val**2 for val, weight in zip(values, weights) for _ in range(weight)
        ]))
        for values in zipped
    ]
    print(rmse_per_index)

    # =================== 3D PLOT ===================
    ordered_algorithms = [algo for algo in aggregated_results_rmse]

    # Reorder parameters so 'Throughput' appears last
    parameters = list(next(iter(aggregated_results_rmse.values())).keys())
    parameters.remove("Throughput")
    parameters = parameters + ['Throughput']

    # Normalize each parameter individually
    param_norms = {}
    for param in parameters:
        values = [abs(aggregated_results_rmse[algo][param]) for algo in ordered_algorithms]
        param_norms[param] = Normalize(vmin=min(values), vmax=max(values))

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    dx = dy = 0.8
    cmap = cm.viridis
    colors_dict = {}

    # Draw bars with per-parameter normalization
    for i, param in enumerate(parameters):
        for j, algo in enumerate(ordered_algorithms):
            raw_value = abs(aggregated_results_rmse[algo][param])
            znorm = param_norms[param](raw_value)

            xpos = j
            ypos = i
            zpos = 0
            dz = max(znorm, 0.01)  # Prevent flat bars

            # Assign unique color to each algorithm
            if algo not in colors_dict:
                colors_dict[algo] = cmap(j / len(ordered_algorithms))

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors_dict[algo], edgecolor='black', alpha=0.9)

    # Customize axes
    ax.set_xticks(np.arange(len(ordered_algorithms)) + dx / 2)
    ax.set_yticks(np.arange(len(parameters)) + dy / 2)
    ax.set_xticklabels(ordered_algorithms, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(parameters, fontsize=10)
    ax.set_zlabel("Normalized Value (per parameter)", fontsize=12)
    ax.set_title("3D Comparison of Algorithms by Parameters", fontsize=14, pad=20)

    plt.tight_layout()
    '''
    # =================== Bar Chart Plot ===================
    # Organizing results by algorithm
    r_dict = {r['Algorithm']: r for r in results}

    num_params = len(analyzed_parameters)
    num_cols = 2  # Split into two vertical sections
    num_rows = math.ceil(num_params / num_cols)  # Calculate needed rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8 * num_rows), constrained_layout=True)

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
    
    plt.savefig("outputs/bar_chart.png", dpi=600, bbox_inches='tight')
    plt.show()'''
    # Organize results by algorithm
    r_dict = {r['Algorithm']: r for r in results}

    num_params = len(analyzed_parameters)

    # Split parameters into two halves
    half = math.ceil(num_params / 2)
    param_groups = [analyzed_parameters[:half], analyzed_parameters[half:]]

    # Loop through each group to create two separate figures
    for group_index, param_group in enumerate(param_groups):
        num_params_group = len(param_group)
        num_cols = 1
        num_rows = math.ceil(num_params_group / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows), constrained_layout=True)

        axes = axes.flatten() if num_params_group > 1 else [axes]

        for i, param in enumerate(param_group):
            values = [d[param] for d in r_dict.values()]
            labels = list(r_dict.keys())

            ax = axes[i]
            ax.bar(labels, values, color=["silver", "gold", "blue", "green", "yellow", "red", "black"], edgecolor='black', linewidth=1.2)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_ylabel(param, fontsize=10)

            # Dynamic y-limit
            if max(values) < 0:
                ax.set_ylim(0, min(values) + min(values) * 0.1)
            else:
                ax.set_ylim(0, max(values) + max(values) * 0.1)

            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

        # Remove any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.savefig(f"outputs/bar_chart_part_{group_index + 1}.png", dpi=600, bbox_inches='tight')
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
        ax.text(WirelessNetwork[0].x_position, WirelessNetwork[0].y_position, WirelessNetwork[0].system_name, color='white', ha='center', va='center', fontsize=10, weight='bold')
    
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
mpmo_saw = MPMO_SAW("MPMO-SAW", analyzed_parameters, weights, directions)
mpmo_saw_hyst = MPMO_SAW("MPMO-SAW-Hysteresis", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
mpmo_saw_ttt = MPMO_SAW("MPMO-SAW-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
mpmo_wpm = MPMO_WPM("MPMO-WPM", analyzed_parameters, weights, directions)
mpmo_wpm_hyst = MPMO_WPM("MPMO-WPM-Hysteresis", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
mpmo_wpm_ttt = MPMO_WPM("MPMO-WPM-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
mpmo_topsis = MPMO_TOPSIS("MPMO-TOPSIS", analyzed_parameters, weights, directions)
mpmo_topsis_hyst = MPMO_TOPSIS("MPMO-TOPSIS-Hysteresis", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
mpmo_topsis_ttt = MPMO_TOPSIS("MPMO-TOPSIS-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
benchmark = BenchmarkMethod("Benchmark", analyzed_parameters, directions)

# Instances of Performance Analysis
p_spmo_max_min_rssi = PerformanceAnalysis("SPMO-MAX-RSSI")
p_spmo_max_min_snr = PerformanceAnalysis("SPMO-MAX-SNR")
p_spmo_pref = PerformanceAnalysis("SPMO-Preference")
p_mpmo_saw = PerformanceAnalysis("MPMO-SAW")
p_mpmo_saw_hyst = PerformanceAnalysis("MPMO-SAW-Hyst")
p_mpmo_saw_ttt = PerformanceAnalysis("MPMO-SAW-TTT")
p_mpmo_wpm = PerformanceAnalysis("MPMO-WPM")
p_mpmo_wpm_hyst = PerformanceAnalysis("MPMO-WPM-Hyst")
p_mpmo_wpm_ttt = PerformanceAnalysis("MPMO-WPM-TTT")
p_mpmo_topsis = PerformanceAnalysis("MPMO-TOPSIS")
p_mpmo_topsis_hyst = PerformanceAnalysis("MPMO-TOPSIS-Hyst")
p_mpmo_topsis_ttt = PerformanceAnalysis("MPMO-TOPSIS-TTT")
p_benchmark = PerformanceAnalysis("Benchmark")

update_position(device_1)
root.mainloop()