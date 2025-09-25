import tkinter as tk
from tkinter import ttk
import sys
import random
import numpy as np
import math
import copy
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
from vhSimulator import MPMO_SAW, MPMO_WPM, MPMO_TOPSIS, MPMO_Fuzzy, MPMO_RMSE, NN_TOPSIS, BenchmarkMethod, WorstScenarioMethod, PerformanceAnalysis


# ==================================== Initial Parameters ====================================
# Map dimension
x_max, y_max = 1000, 1000

# Start position
x, y = x_max/2, y_max/2

# Device Average Velocity
device_velocity = 10

# Interval between iterations
iter_interval = 1

# Distance for iteration, 0.1 because the iter_interval is 100ms
dist_iter = device_velocity * 0.1

# n = Number of iterations, j = DO NOT CHANGE
j = 0
n = 500

# Activate Graphical Interface
GUI = False

# Activate Prints for DEBBUG
verbose = False

# Number of simulations
n_simulations = 50

# Iteration x Simulations
iter_x_simu = n_simulations * n

# Plot Results
plots = True

# Predef Configs
predef_conf = False

# Corrections Real World Applications
corrections_real_world = True

# Fading [None, "Rayleigh", "Rician"]
fading = "Rician"

# Performance Analysis
analyzed_parameters = ['RSSI', 'SNR', 'Throughput', 'PC', 'MC', 'BER', 'FEC']
weights = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
weights_total_sum = 1
directions = [1, 1, 1, 0, 0, 0, 1]
lockin_percentage = 0.03
tt_trigger = 3
hyst_percentage = 0.3

# Results
final_results = []
indicators_results = []

# Random Walk
random_walk_times = 20

# DO NOT CHANGE
random_direction_counter = 0
old_dx = 0
old_dy = 0

# To delete
count_nn = 0
# ============================================================================================




# ===================================== MADM Algorithms ======================================
SPMO_Methods = True

SAW = True

WPM = True

TOPSIS = True

Fuzzy = True

RMSE = True

TOPSIS_NN = False
# ============================================================================================




# ===================================== Improviment Techniques ======================================
impTech_hyst = False

impTech_lockin = False

impTech_TTT = False
# ============================================================================================



# Wireless Network Systems
WNS_list = []

def generate_random_WNS(predef):
    global WNS_list
    WNS_list = []
    
    # WiFi's
    global wifi_1
    wifi_1 = WNS("WiFi-1", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_1, 'green', 0.3])
    
    global wifi_2
    wifi_2 = WNS("WiFi-2", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 7, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_2, 'green', 0.3])
    
    global wifi_3
    wifi_3 = WNS("WiFi-3", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 7, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_3, 'green', 0.3])
    
    global wifi_4
    wifi_4 = WNS("WiFi-4", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 7, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_4, 'green', 0.3])
    
    global wifi_5
    wifi_5 = WNS("WiFi-5", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 7, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_5, 'green', 0.3])
    
    global wifi_6
    wifi_6 = WNS("WiFi-6", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_6, 'green', 0.3])
    
    global wifi_7
    wifi_7 = WNS("WiFi-7", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_7, 'green', 0.3])

    global wifi_8
    wifi_8 = WNS("WiFi-8", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_8, 'green', 0.3])
    
    global wifi_9
    wifi_9 = WNS("WiFi-9", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 7, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 150000000], predef_snr=[40, 15], predef_rssi=[-50, -80], predef_ber=[0.00000001, 0.000001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_9, 'green', 0.3])
    
    global wifi_10
    wifi_10 = WNS("WiFi-10", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 30000000], predef_snr=[40, 10], predef_rssi=[-50, -80], predef_ber=[0.000001, 0.0001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_10, 'green', 0.3])

    
    # NB-IoT
    global nbiot_1
    nbiot_1 = WNS("NBIoT-1", random.uniform(-6*x_max, 6*x_max), random.uniform(-6*y_max, 6*y_max), 30, 800000000, 1400000, 2, "NB-IoT", 0.25, 5, maximum_radius=15000, predef_throughput=[100000, 10000], predef_snr=[10, 2], predef_rssi=[-90, -115], predef_ber=[0.00001, 0.001], predef_fec=[2/3, 1/3], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([nbiot_1, 'blue', 0.03])


    # LoRa's
    global LoRa_1
    LoRa_1 = WNS("LoRa-1", random.uniform(-5*x_max, 5*x_max), random.uniform(-5*y_max, 5*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 1000], predef_snr=[10, 0], predef_rssi=[-80, -120], predef_ber=[0.00001, 0.01], predef_fec=[4/5, 4/8], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([LoRa_1, 'yellow', 0.03])
    
    global LoRa_2
    LoRa_2 = WNS("LoRa-2", random.uniform(-5*x_max, 5*x_max), random.uniform(-5*y_max, 5*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 1000], predef_snr=[10, 0], predef_rssi=[-80, -120], predef_ber=[0.00001, 0.01], predef_fec=[4/5, 4/8], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([LoRa_2, 'yellow', 0.03])


    # LTE 4G
    global LTE_4g
    LTE_4g = WNS("LTE-4g-1", random.uniform(-8*x_max, 8*x_max), random.uniform(-8*y_max, 8*y_max), 40, 1900000000, 20000000, 5, "LTE-4G", 1.05, 3, maximum_radius=30000, predef_throughput=[100000000, 5000000], predef_snr=[15, 5], predef_rssi=[-70, -100], predef_ber=[0.000001, 0.0001], predef_fec=[3/4, 1/3], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([LTE_4g, 'red', 0.03])


    # WiMax
    global wifi_max_1
    wifi_max_1 = WNS("WiMax-1", random.uniform(-4*x_max, 4*x_max), random.uniform(-4*y_max, 4*y_max), 40, 3000000000, 10000000, 10, "WiMax", 0.80, 2, maximum_radius=6000, predef_throughput=[40000000, 2000000], predef_snr=[15, 5], predef_rssi=[-60, -90], predef_ber=[0.0000001, 0.00001], predef_fec=[5/6, 1/2], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([wifi_max_1, 'purple', 0.03])
    
    
    # 5G NR
    '''
    global nr_5g_1
    nr_5g_1 = WNS("5G-NR", random.uniform(-2*x_max, 2*x_max), random.uniform(-2*y_max, 2*y_max), 43, 3500000000, 100000000, 10, "5G-NR", 1.5, 15, maximum_radius=1000, predef_throughput=[1000000000, 100000000], predef_snr=[20, 10], predef_rssi=[-65, -85], predef_ber=[1e-6, 1e-4], predef_fec=[0.9, 0.75], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([nr_5g_1, 'brown', 0.03])
    
    global nr_5g_2
    nr_5g_2 = WNS("5G-NR", random.uniform(-2*x_max, 2*x_max), random.uniform(-2*y_max, 2*y_max), 43, 3500000000, 100000000, 10, "5G-NR", 1.5, 15, maximum_radius=1000, predef_throughput=[1000000000, 100000000], predef_snr=[20, 10], predef_rssi=[-65, -85], predef_ber=[1e-6, 1e-4], predef_fec=[0.9, 0.75], predef_config=predef, corrections_real_world_applications=corrections_real_world, fading=fading)
    WNS_list.append([nr_5g_2, 'brown', 0.03])
    '''
    
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
    device.connect_to_network(nbiot_1)
    device.connect_to_network(LoRa_1)
    device.connect_to_network(LoRa_2)
    device.connect_to_network(LTE_4g)
    device.connect_to_network(wifi_max_1)
    #device.connect_to_network(nr_5g_1)
    #device.connect_to_network(nr_5g_2)



# =============================== Initialize Graph ===============================

def random_direction():
    global dist_iter
    # Random angle in radians
    angle = random.uniform(0, 2 * np.pi)
    dx = dist_iter * np.cos(angle)
    dy = dist_iter * np.sin(angle)
    return dx, dy


def update_position(device):
    global x, y, j, n, iter_interval, WNS_list, n_simulations, x_max, y_max, predef_conf, random_direction_counter, random_walk_times, old_dx, old_dy
    if random_direction_counter == 0 or (random_direction_counter%random_walk_times) == 0:
        dx, dy = random_direction()
        old_dx = dx
        old_dy = dy
    else:
        dx, dy = old_dx, old_dy
    x = min(max(x + dx, 0), x_max)
    y = min(max(y + dy, 0), y_max)
    random_direction_counter += 1
    
    calculate_parameters(device, x, y)
    
    if GUI == True: plot_graph()
    
    j += 1
    if j < n:
        root.after(iter_interval, lambda: update_position(device))
    else:
        performe_analysis()
        n_simulations = n_simulations - 1
        if n_simulations != 0:
            x = x_max/2
            y = y_max/2
            device = Device(1, x, y)
            generate_random_WNS(predef_conf)
            connect_to_net(device)
            # Cleaning old QoS parameters Storaged
            if SPMO_Methods == True:
                p_spmo_max_min_rssi.clean_storaged_QoS()
                p_spmo_max_min_snr.clean_storaged_QoS()
                p_spmo_pref.clean_storaged_QoS()
            if SAW == True:
                p_mpmo_saw.clean_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_saw_lockin.clean_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_saw_hyst.clean_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_saw_ttt.clean_storaged_QoS()
            if WPM == True:
                p_mpmo_wpm.clean_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_wpm_hyst.clean_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_wpm_lockin.clean_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_wpm_ttt.clean_storaged_QoS()
            if TOPSIS == True:
                p_mpmo_topsis.clean_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_topsis_hyst.clean_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_topsis_lockin.clean_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_topsis_ttt.clean_storaged_QoS()
            if Fuzzy == True:
                p_mpmo_fuzzy.clean_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_fuzzy_hyst.clean_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_fuzzy_lockin.clean_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_fuzzy_ttt.clean_storaged_QoS()
            if RMSE == True:
                p_mpmo_rmse.clean_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_rmse_hyst.clean_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_rmse_lockin.clean_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_rmse_ttt.clean_storaged_QoS()
            if TOPSIS_NN == True:
                p_nn_topsis.clean_storaged_QoS()
            p_benchmark.clean_storaged_QoS()
            p_worst_scenario.clean_storaged_QoS()
            # Cleaning old Benchmark QoS parameters Storaged
            if SPMO_Methods == True:
                p_spmo_max_min_rssi.clean_Benchmark_storaged_QoS()
                p_spmo_max_min_snr.clean_Benchmark_storaged_QoS()
                p_spmo_pref.clean_Benchmark_storaged_QoS()
            if SAW == True:
                p_mpmo_saw.clean_Benchmark_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_saw_hyst.clean_Benchmark_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_saw_lockin.clean_Benchmark_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_saw_ttt.clean_Benchmark_storaged_QoS()
            if WPM == True:
                p_mpmo_wpm.clean_Benchmark_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_wpm_hyst.clean_Benchmark_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_wpm_lockin.clean_Benchmark_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_wpm_ttt.clean_Benchmark_storaged_QoS()
            if TOPSIS == True:
                p_mpmo_topsis.clean_Benchmark_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_topsis_hyst.clean_Benchmark_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_topsis_lockin.clean_Benchmark_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_topsis_ttt.clean_Benchmark_storaged_QoS()
            if Fuzzy == True:
                p_mpmo_fuzzy.clean_Benchmark_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_fuzzy_hyst.clean_Benchmark_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_fuzzy_lockin.clean_Benchmark_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_fuzzy_ttt.clean_Benchmark_storaged_QoS()
            if RMSE == True:
                p_mpmo_rmse.clean_Benchmark_storaged_QoS()
                if impTech_hyst == True:
                    p_mpmo_rmse_hyst.clean_Benchmark_storaged_QoS()
                if impTech_lockin == True:
                    p_mpmo_rmse_lockin.clean_Benchmark_storaged_QoS()
                if impTech_TTT == True:
                    p_mpmo_rmse_ttt.clean_Benchmark_storaged_QoS()
            if TOPSIS_NN == True:
                p_nn_topsis.clean_Benchmark_storaged_QoS()
            p_benchmark.clean_Benchmark_storaged_QoS()
            p_worst_scenario.clean_Benchmark_storaged_QoS()
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
    if predef_conf == True:
        device.get_all_QoS_Parameters_predef()
    else:
        device.get_all_QoS_Parameters()
    available_networks = device.get_available_networks()
    if verbose == True: print(f"Available Networks: {available_networks}")
    
    
    # ============================== Benchmark  ==============================
    if available_networks != []:
        decision_benchmark = device.makeDecision(benchmark, available_networks)
    else:
        decision_benchmark = {'Network': 'Benchmark', 'Status': 'Offline', 'Protocol': 'Benchmark', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
    p_benchmark.store_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Benchmark {decision_benchmark}")
        
    # ============================== Worst Scenario ==============================
    if available_networks != []:
        decision_worst_scenario = device.makeDecision(worst_scenario, available_networks)
    else:
        decision_worst_scenario = {'Network': 'WorstScenario', 'Status': 'Offline', 'Protocol': 'WorstScenario', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
    p_worst_scenario.store_QoS_parameters(decision_worst_scenario)
    p_worst_scenario.store_Benchmark_QoS_parameters(decision_benchmark)
    if verbose == True: print(f"Worst Scenario: {decision_worst_scenario}")
    
    if SPMO_Methods == True:
        if available_networks != []:
            decision_spmo_mmm_rssi = device.makeDecision(spmo_max_min_method_rssi, available_networks)
        else:
            decision_spmo_mmm_rssi = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_spmo_max_min_rssi.store_QoS_parameters(decision_spmo_mmm_rssi)
        p_spmo_max_min_rssi.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision SPMO MAX MIN RSSI: {decision_spmo_mmm_rssi}")
        
        if available_networks != []:
            decision_spmo_mmm_snr = device.makeDecision(spmo_max_min_method_snr, available_networks)
        else:
            decision_spmo_mmm_snr = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_spmo_max_min_snr.store_QoS_parameters(decision_spmo_mmm_snr)
        p_spmo_max_min_snr.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision SPMO MAX MIN SNR: {decision_spmo_mmm_snr}")
        
        if available_networks != []:
            decision_spmo_pref = device.makeDecision(spmo_pref, available_networks)
        else:
            decision_spmo_pref = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_spmo_pref.store_QoS_parameters(decision_spmo_pref)
        p_spmo_pref.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision SPMO Preference: {decision_spmo_pref}")
    
    if SAW == True:
        if available_networks != []:
            decision_mpmo_saw = device.makeDecision(mpmo_saw, available_networks)
        else:
            decision_mpmo_saw = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_mpmo_saw.store_QoS_parameters(decision_mpmo_saw)
        p_mpmo_saw.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision MPMO SAW: {decision_mpmo_saw}")
        
        if impTech_hyst == True:
            if available_networks != []:
                decision_mpmo_saw_hyst = device.makeDecision(mpmo_saw_hyst, available_networks)
            else:
                decision_mpmo_saw_hyst = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_saw_hyst.store_QoS_parameters(decision_mpmo_saw_hyst)
            p_mpmo_saw_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO SAW Hysteresis: {decision_mpmo_saw_hyst}")
        
        if impTech_lockin == True:
            if available_networks != []:
                decision_mpmo_saw_lockin = device.makeDecision(mpmo_saw_lockin, available_networks)
            else:
                decision_mpmo_saw_lockin = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_saw_lockin.store_QoS_parameters(decision_mpmo_saw_lockin)
            p_mpmo_saw_lockin.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO SAW LockIn: {decision_mpmo_saw_lockin}")
        
        if impTech_TTT == True:
            if available_networks != []:
                decision_mpmo_saw_ttt = device.makeDecision(mpmo_saw_ttt, available_networks)
            else:
                decision_mpmo_saw_ttt = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_saw_ttt.store_QoS_parameters(decision_mpmo_saw_ttt)
            p_mpmo_saw_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO SAW Time to Trigger: {decision_mpmo_saw_ttt}")
    
    if WPM == True:
        if available_networks != []:
            decision_mpmo_wpm = device.makeDecision(mpmo_wpm, available_networks)
        else:
            decision_mpmo_wpm = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_mpmo_wpm.store_QoS_parameters(decision_mpmo_wpm)
        p_mpmo_wpm.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision MPMO WPM: {decision_mpmo_wpm}")
        
        if impTech_hyst == True:
            if available_networks != []:
                decision_mpmo_wpm_hyst = device.makeDecision(mpmo_wpm_hyst, available_networks)
            else:
                decision_mpmo_wpm_hyst = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_wpm_hyst.store_QoS_parameters(decision_mpmo_wpm_hyst)
            p_mpmo_wpm_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO WPM Hyst: {decision_mpmo_wpm_hyst}")
        
        if impTech_lockin == True:
            if available_networks != []:
                decision_mpmo_wpm_lockin = device.makeDecision(mpmo_wpm_lockin, available_networks)
            else:
                decision_mpmo_wpm_lockin = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_wpm_lockin.store_QoS_parameters(decision_mpmo_wpm_lockin)
            p_mpmo_wpm_lockin.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO WPM LockIn: {decision_mpmo_wpm_lockin}")
        
        if impTech_TTT == True:
            if available_networks != []:
                decision_mpmo_wpm_ttt = device.makeDecision(mpmo_wpm_ttt, available_networks)
            else:
                decision_mpmo_wpm_ttt = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_wpm_ttt.store_QoS_parameters(decision_mpmo_wpm_ttt)
            p_mpmo_wpm_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO WPM Time to Trigger: {decision_mpmo_wpm_ttt}")
    
    if TOPSIS == True:
        if available_networks != []:
            decision_mpmo_topsis = device.makeDecision(mpmo_topsis, available_networks)
        else:
            decision_mpmo_topsis = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_mpmo_topsis.store_QoS_parameters(decision_mpmo_topsis)
        p_mpmo_topsis.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision MPMO TOPSIS: {decision_mpmo_topsis}")
        
        if impTech_hyst == True:
            if available_networks != []:
                decision_mpmo_topsis_hyst = device.makeDecision(mpmo_topsis_hyst, available_networks)
            else:
                decision_mpmo_topsis_hyst = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_topsis_hyst.store_QoS_parameters(decision_mpmo_topsis_hyst)
            p_mpmo_topsis_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO TOPSIS Hyst: {decision_mpmo_topsis_hyst}")
        
        if impTech_lockin == True:
            if available_networks != []:
                decision_mpmo_topsis_lockin = device.makeDecision(mpmo_topsis_lockin, available_networks)
            else:
                decision_mpmo_topsis_lockin = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_topsis_lockin.store_QoS_parameters(decision_mpmo_topsis_lockin)
            p_mpmo_topsis_lockin.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO TOPSIS LockIn: {decision_mpmo_topsis_lockin}")
        
        if impTech_TTT == True:
            if available_networks != []:
                decision_mpmo_topsis_ttt = device.makeDecision(mpmo_topsis_ttt, available_networks)
            else:
                decision_mpmo_topsis_ttt = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_topsis_ttt.store_QoS_parameters(decision_mpmo_topsis_ttt)
            p_mpmo_topsis_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO TOPSIS Time to Trigger: {decision_mpmo_topsis_ttt}")
    
    if Fuzzy == True:
        
        if available_networks != []:
            decision_mpmo_fuzzy = device.makeDecision(mpmo_fuzzy, available_networks)
        else:
            decision_mpmo_fuzzy = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_mpmo_fuzzy.store_QoS_parameters(decision_mpmo_fuzzy)
        p_mpmo_fuzzy.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision MPMO Fuzzy: {decision_mpmo_fuzzy}")
        
        if impTech_hyst == True:
            if available_networks != []:
                decision_mpmo_fuzzy_hyst = device.makeDecision(mpmo_fuzzy_hyst, available_networks)
            else:
                decision_mpmo_fuzzy_hyst = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_fuzzy_hyst.store_QoS_parameters(decision_mpmo_fuzzy_hyst)
            p_mpmo_fuzzy_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO Fuzzy Hyst: {decision_mpmo_fuzzy_hyst}")
        
        if impTech_lockin == True:
            if available_networks != []:
                decision_mpmo_fuzzy_lockin = device.makeDecision(mpmo_fuzzy_lockin, available_networks)
            else:
                decision_mpmo_fuzzy_lockin = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_fuzzy_lockin.store_QoS_parameters(decision_mpmo_fuzzy_lockin)
            p_mpmo_fuzzy_lockin.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO Fuzzy LockIn: {decision_mpmo_fuzzy_lockin}")
        
        if impTech_TTT == True:
            if available_networks != []:
                decision_mpmo_fuzzy_ttt = device.makeDecision(mpmo_fuzzy_ttt, available_networks)
            else:
                decision_mpmo_fuzzy_ttt = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_fuzzy_ttt.store_QoS_parameters(decision_mpmo_fuzzy_ttt)
            p_mpmo_fuzzy_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO Fuzzy Time to Trigger: {decision_mpmo_fuzzy_ttt}")
    
    if RMSE == True:
        
        if available_networks != []:
            decision_mpmo_rmse = device.makeDecision(mpmo_rmse, available_networks)
        else:
            decision_mpmo_rmse = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_mpmo_rmse.store_QoS_parameters(decision_mpmo_rmse)
        p_mpmo_rmse.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision MPMO RMSE: {decision_mpmo_rmse}")
        
        if impTech_hyst == True:
            if available_networks != []:
                decision_mpmo_rmse_hyst = device.makeDecision(mpmo_rmse_hyst, available_networks)
            else:
                decision_mpmo_rmse_hyst = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_rmse_hyst.store_QoS_parameters(decision_mpmo_rmse_hyst)
            p_mpmo_rmse_hyst.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO RMSE Hyst: {decision_mpmo_rmse_hyst}")
        
        if impTech_lockin == True:
            if available_networks != []:
                decision_mpmo_rmse_lockin = device.makeDecision(mpmo_rmse_lockin, available_networks)
            else:
                decision_mpmo_rmse_lockin = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_rmse_lockin.store_QoS_parameters(decision_mpmo_rmse_lockin)
            p_mpmo_rmse_lockin.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO RMSE LockIn: {decision_mpmo_rmse_lockin}")
        
        if impTech_TTT == True:
            if available_networks != []:
                decision_mpmo_rmse_ttt = device.makeDecision(mpmo_rmse_ttt, available_networks)
            else:
                decision_mpmo_rmse_ttt = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
            p_mpmo_rmse_ttt.store_QoS_parameters(decision_mpmo_rmse_ttt)
            p_mpmo_rmse_ttt.store_Benchmark_QoS_parameters(decision_benchmark)
            if verbose == True: print(f"Decision MPMO RMSE Time to Trigger: {decision_mpmo_rmse_ttt}")
    
    if TOPSIS_NN == True:
    
        if available_networks != []:
            decision_nn_topsis = device.makeDecision(nn_topsis, available_networks)
        else:
            decision_nn_topsis = {'Network': 'Offline', 'Status': 'Offline', 'Protocol': 'Offline', 'RSSI': 0, 'SNR': 0, 'Throughput': 0, 'PC': 0, 'MC': 0, 'BER': 0, 'FEC': 0}
        p_nn_topsis.store_QoS_parameters(decision_nn_topsis)
        p_nn_topsis.store_Benchmark_QoS_parameters(decision_benchmark)
        if verbose == True: print(f"Decision NN TOPSIS: {decision_nn_topsis}")
    
    #global count_nn
    #if decision_nn_topsis == decision_mpmo_topsis: count_nn = count_nn + 1
    #print(count_nn)
    
    
    if verbose == True: print("===================================================")


def performe_analysis():
    global final_results, analyzed_parameters, indicators_results
    
    # Gathering together the results
    results_list = []
    indicators_list = []
    
    if SPMO_Methods == True:
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
    
    if SAW == True:
        results_mpmo_saw = p_mpmo_saw.calculate_average_QoS_parameters(analyzed_parameters)
        results_mpmo_saw['Handover'] = p_mpmo_saw.count_number_of_handovers()
        results_mpmo_saw['Algorithm'] = p_mpmo_saw.algorithm
        indicators_mpmo_saw = p_mpmo_saw.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_mpmo_saw['Algorithm'] = results_mpmo_saw['Algorithm']
        indicators_list.append(indicators_mpmo_saw)
        results_list.append(results_mpmo_saw)
        
        if impTech_hyst == True:
            results_mpmo_saw_hyst = p_mpmo_saw_hyst.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_saw_hyst['Handover'] = p_mpmo_saw_hyst.count_number_of_handovers()
            results_mpmo_saw_hyst['Algorithm'] = p_mpmo_saw_hyst.algorithm
            indicators_mpmo_saw_hyst = p_mpmo_saw_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_saw_hyst['Algorithm'] = results_mpmo_saw_hyst['Algorithm']
            indicators_list.append(indicators_mpmo_saw_hyst)
            results_list.append(results_mpmo_saw_hyst)
        
        if impTech_lockin == True:
            results_mpmo_saw_lockin = p_mpmo_saw_lockin.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_saw_lockin['Handover'] = p_mpmo_saw_lockin.count_number_of_handovers()
            results_mpmo_saw_lockin['Algorithm'] = p_mpmo_saw_lockin.algorithm
            indicators_mpmo_saw_lockin = p_mpmo_saw_lockin.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_saw_lockin['Algorithm'] = results_mpmo_saw_lockin['Algorithm']
            indicators_list.append(indicators_mpmo_saw_lockin)
            results_list.append(results_mpmo_saw_lockin)
        
        if impTech_TTT == True:
            results_mpmo_saw_ttt = p_mpmo_saw_ttt.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_saw_ttt['Handover'] = p_mpmo_saw_ttt.count_number_of_handovers()
            results_mpmo_saw_ttt['Algorithm'] = p_mpmo_saw_ttt.algorithm
            indicators_mpmo_saw_ttt = p_mpmo_saw_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_saw_ttt['Algorithm'] = results_mpmo_saw_ttt['Algorithm']
            indicators_list.append(indicators_mpmo_saw_ttt)
            results_list.append(results_mpmo_saw_ttt)
    
    if WPM == True:
        results_mpmo_wpm = p_mpmo_wpm.calculate_average_QoS_parameters(analyzed_parameters)
        results_mpmo_wpm['Handover'] = p_mpmo_wpm.count_number_of_handovers()
        results_mpmo_wpm['Algorithm'] = p_mpmo_wpm.algorithm
        indicators_mpmo_wpm = p_mpmo_wpm.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_mpmo_wpm['Algorithm'] = results_mpmo_wpm['Algorithm']
        indicators_list.append(indicators_mpmo_wpm)
        results_list.append(results_mpmo_wpm)
        
        if impTech_hyst == True:
            results_mpmo_wpm_hyst = p_mpmo_wpm_hyst.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_wpm_hyst['Handover'] = p_mpmo_wpm_hyst.count_number_of_handovers()
            results_mpmo_wpm_hyst['Algorithm'] = p_mpmo_wpm_hyst.algorithm
            indicators_mpmo_wpm_hyst = p_mpmo_wpm_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_wpm_hyst['Algorithm'] = results_mpmo_wpm_hyst['Algorithm']
            indicators_list.append(indicators_mpmo_wpm_hyst)
            results_list.append(results_mpmo_wpm_hyst)
        
        if impTech_lockin == True:
            results_mpmo_wpm_lockin = p_mpmo_wpm_lockin.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_wpm_lockin['Handover'] = p_mpmo_wpm_lockin.count_number_of_handovers()
            results_mpmo_wpm_lockin['Algorithm'] = p_mpmo_wpm_lockin.algorithm
            indicators_mpmo_wpm_lockin = p_mpmo_wpm_lockin.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_wpm_lockin['Algorithm'] = results_mpmo_wpm_lockin['Algorithm']
            indicators_list.append(indicators_mpmo_wpm_lockin)
            results_list.append(results_mpmo_wpm_lockin)
        
        if impTech_TTT == True:
            results_mpmo_wpm_ttt = p_mpmo_wpm_ttt.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_wpm_ttt['Handover'] = p_mpmo_wpm_ttt.count_number_of_handovers()
            results_mpmo_wpm_ttt['Algorithm'] = p_mpmo_wpm_ttt.algorithm
            indicators_mpmo_wpm_ttt = p_mpmo_wpm_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_wpm_ttt['Algorithm'] = results_mpmo_wpm_ttt['Algorithm']
            indicators_list.append(indicators_mpmo_wpm_ttt)
            results_list.append(results_mpmo_wpm_ttt)
    
    if TOPSIS == True:
        results_mpmo_topsis = p_mpmo_topsis.calculate_average_QoS_parameters(analyzed_parameters)
        results_mpmo_topsis['Handover'] = p_mpmo_topsis.count_number_of_handovers()
        results_mpmo_topsis['Algorithm'] = p_mpmo_topsis.algorithm
        indicators_mpmo_topsis = p_mpmo_topsis.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_mpmo_topsis['Algorithm'] = results_mpmo_topsis['Algorithm']
        indicators_list.append(indicators_mpmo_topsis)
        results_list.append(results_mpmo_topsis)
        
        if impTech_hyst == True:
            results_mpmo_topsis_hyst = p_mpmo_topsis_hyst.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_topsis_hyst['Handover'] = p_mpmo_topsis_hyst.count_number_of_handovers()
            results_mpmo_topsis_hyst['Algorithm'] = p_mpmo_topsis_hyst.algorithm
            indicators_mpmo_topsis_hyst = p_mpmo_topsis_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_topsis_hyst['Algorithm'] = results_mpmo_topsis_hyst['Algorithm']
            indicators_list.append(indicators_mpmo_topsis_hyst)
            results_list.append(results_mpmo_topsis_hyst)
        
        if impTech_lockin == True:
            results_mpmo_topsis_lockin = p_mpmo_topsis_lockin.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_topsis_lockin['Handover'] = p_mpmo_topsis_lockin.count_number_of_handovers()
            results_mpmo_topsis_lockin['Algorithm'] = p_mpmo_topsis_lockin.algorithm
            indicators_mpmo_topsis_lockin = p_mpmo_topsis_lockin.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_topsis_lockin['Algorithm'] = results_mpmo_topsis_lockin['Algorithm']
            indicators_list.append(indicators_mpmo_topsis_lockin)
            results_list.append(results_mpmo_topsis_lockin)
        
        if impTech_TTT == True:
            results_mpmo_topsis_ttt = p_mpmo_topsis_ttt.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_topsis_ttt['Handover'] = p_mpmo_topsis_ttt.count_number_of_handovers()
            results_mpmo_topsis_ttt['Algorithm'] = p_mpmo_topsis_ttt.algorithm
            indicators_mpmo_topsis_ttt = p_mpmo_topsis_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_topsis_ttt['Algorithm'] = results_mpmo_topsis_ttt['Algorithm']
            indicators_list.append(indicators_mpmo_topsis_ttt)
            results_list.append(results_mpmo_topsis_ttt)
    
    if Fuzzy == True:
        results_mpmo_fuzzy = p_mpmo_fuzzy.calculate_average_QoS_parameters(analyzed_parameters)
        results_mpmo_fuzzy['Handover'] = p_mpmo_fuzzy.count_number_of_handovers()
        results_mpmo_fuzzy['Algorithm'] = p_mpmo_fuzzy.algorithm
        indicators_mpmo_fuzzy = p_mpmo_fuzzy.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_mpmo_fuzzy['Algorithm'] = results_mpmo_fuzzy['Algorithm']
        indicators_list.append(indicators_mpmo_fuzzy)
        results_list.append(results_mpmo_fuzzy)
        
        if impTech_hyst == True:
            results_mpmo_fuzzy_hyst = p_mpmo_fuzzy_hyst.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_fuzzy_hyst['Handover'] = p_mpmo_fuzzy_hyst.count_number_of_handovers()
            results_mpmo_fuzzy_hyst['Algorithm'] = p_mpmo_fuzzy_hyst.algorithm
            indicators_mpmo_fuzzy_hyst = p_mpmo_fuzzy_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_fuzzy_hyst['Algorithm'] = results_mpmo_fuzzy_hyst['Algorithm']
            indicators_list.append(indicators_mpmo_fuzzy_hyst)
            results_list.append(results_mpmo_fuzzy_hyst)
        
        if impTech_lockin == True:
            results_mpmo_fuzzy_lockin = p_mpmo_fuzzy_lockin.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_fuzzy_lockin['Handover'] = p_mpmo_fuzzy_lockin.count_number_of_handovers()
            results_mpmo_fuzzy_lockin['Algorithm'] = p_mpmo_fuzzy_lockin.algorithm
            indicators_mpmo_fuzzy_lockin = p_mpmo_fuzzy_lockin.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_fuzzy_lockin['Algorithm'] = results_mpmo_fuzzy_lockin['Algorithm']
            indicators_list.append(indicators_mpmo_fuzzy_lockin)
            results_list.append(results_mpmo_fuzzy_lockin)
        
        if impTech_TTT == True:
            results_mpmo_fuzzy_ttt = p_mpmo_fuzzy_ttt.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_fuzzy_ttt['Handover'] = p_mpmo_fuzzy_ttt.count_number_of_handovers()
            results_mpmo_fuzzy_ttt['Algorithm'] = p_mpmo_fuzzy_ttt.algorithm
            indicators_mpmo_fuzzy_ttt = p_mpmo_fuzzy_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_fuzzy_ttt['Algorithm'] = results_mpmo_fuzzy_ttt['Algorithm']
            indicators_list.append(indicators_mpmo_fuzzy_ttt)
            results_list.append(results_mpmo_fuzzy_ttt)
    
    if RMSE == True:
        results_mpmo_rmse = p_mpmo_rmse.calculate_average_QoS_parameters(analyzed_parameters)
        results_mpmo_rmse['Handover'] = p_mpmo_rmse.count_number_of_handovers()
        results_mpmo_rmse['Algorithm'] = p_mpmo_rmse.algorithm
        indicators_mpmo_rsme = p_mpmo_rmse.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_mpmo_rsme['Algorithm'] = results_mpmo_rmse['Algorithm']
        indicators_list.append(indicators_mpmo_rsme)
        results_list.append(results_mpmo_rmse)
        
        if impTech_hyst == True:
            results_mpmo_rmse_hyst = p_mpmo_rmse_hyst.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_rmse_hyst['Handover'] = p_mpmo_rmse_hyst.count_number_of_handovers()
            results_mpmo_rmse_hyst['Algorithm'] = p_mpmo_rmse_hyst.algorithm
            indicators_mpmo_rsme_hyst = p_mpmo_rmse_hyst.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_rsme_hyst['Algorithm'] = results_mpmo_rmse_hyst['Algorithm']
            indicators_list.append(indicators_mpmo_rsme_hyst)
            results_list.append(results_mpmo_rmse_hyst)
        
        if impTech_lockin == True:
            results_mpmo_rmse_lockin = p_mpmo_rmse_lockin.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_rmse_lockin['Handover'] = p_mpmo_rmse_lockin.count_number_of_handovers()
            results_mpmo_rmse_lockin['Algorithm'] = p_mpmo_rmse_lockin.algorithm
            indicators_mpmo_rsme_lockin = p_mpmo_rmse_lockin.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_rsme_lockin['Algorithm'] = results_mpmo_rmse_lockin['Algorithm']
            indicators_list.append(indicators_mpmo_rsme_lockin)
            results_list.append(results_mpmo_rmse_lockin)
        
        if impTech_TTT == True:
            results_mpmo_rmse_ttt = p_mpmo_rmse_ttt.calculate_average_QoS_parameters(analyzed_parameters)
            results_mpmo_rmse_ttt['Handover'] = p_mpmo_rmse_ttt.count_number_of_handovers()
            results_mpmo_rmse_ttt['Algorithm'] = p_mpmo_rmse_ttt.algorithm
            indicators_mpmo_rsme_ttt = p_mpmo_rmse_ttt.calculate_Abs_error_QoS_parameters(analyzed_parameters)
            indicators_mpmo_rsme_ttt['Algorithm'] = results_mpmo_rmse_ttt['Algorithm']
            indicators_list.append(indicators_mpmo_rsme_ttt)
            results_list.append(results_mpmo_rmse_ttt)
    
    if TOPSIS_NN == True: 
        results_nn_topsis = p_nn_topsis.calculate_average_QoS_parameters(analyzed_parameters)
        results_nn_topsis['Handover'] = p_nn_topsis.count_number_of_handovers()
        results_nn_topsis['Algorithm'] = p_nn_topsis.algorithm
        indicators_nn_topsis = p_nn_topsis.calculate_Abs_error_QoS_parameters(analyzed_parameters)
        indicators_nn_topsis['Algorithm'] = results_nn_topsis['Algorithm']
        indicators_list.append(indicators_nn_topsis)
        results_list.append(results_nn_topsis)
    
    indicators_worst_scenario = p_worst_scenario.calculate_Abs_error_QoS_parameters(analyzed_parameters)
    indicators_worst_scenario['Algorithm'] = p_worst_scenario.algorithm
    indicators_list.append(indicators_worst_scenario)
    
    results_benchmark = p_benchmark.calculate_average_QoS_parameters(analyzed_parameters)
    results_benchmark['Handover'] = p_benchmark.count_number_of_handovers()
    results_benchmark['Algorithm'] = p_benchmark.algorithm
    results_list.append(results_benchmark)
    
    
    # Print the Results
    if SPMO_Methods == True:
        print(f"{results_spmo_max_min_rssi}, Handoff: {results_spmo_max_min_rssi['Handover']}")
        print(f"{results_spmo_max_min_snr}, Handoff: {results_spmo_max_min_snr['Handover']}")
        print(f"{results_spmo_pref}, Handoff: {results_spmo_pref['Handover']}")
    if SAW == True:
        print(f"{results_mpmo_saw}, Handoff: {results_mpmo_saw['Handover']}")
        if impTech_hyst == True:
            print(f"{results_mpmo_saw_hyst}, Handoff: {results_mpmo_saw_hyst['Handover']}")
        if impTech_lockin == True:
            print(f"{results_mpmo_saw_lockin}, Handoff: {results_mpmo_saw_lockin['Handover']}")
        if impTech_TTT == True:
            print(f"{results_mpmo_saw_ttt}, Handoff: {results_mpmo_saw_ttt['Handover']}")
    if WPM == True:
        print(f"{results_mpmo_wpm}, Handoff: {results_mpmo_wpm['Handover']}")
        if impTech_hyst == True:
            print(f"{results_mpmo_wpm_hyst}, Handoff: {results_mpmo_wpm_hyst['Handover']}")
        if impTech_lockin == True:
            print(f"{results_mpmo_wpm_lockin}, Handoff: {results_mpmo_wpm_lockin['Handover']}")
        if impTech_TTT == True:
            print(f"{results_mpmo_wpm_ttt}, Handoff: {results_mpmo_wpm_ttt['Handover']}")
    if TOPSIS == True:
        print(f"{results_mpmo_topsis}, Handoff: {results_mpmo_topsis['Handover']}")
        if impTech_hyst == True:
            print(f"{results_mpmo_topsis_hyst}, Handoff: {results_mpmo_topsis_hyst['Handover']}")
        if impTech_lockin == True:
            print(f"{results_mpmo_topsis_lockin}, Handoff: {results_mpmo_topsis_lockin['Handover']}")
        if impTech_TTT == True:
            print(f"{results_mpmo_topsis_ttt}, Handoff: {results_mpmo_topsis_ttt['Handover']}")
    if Fuzzy == True:
        print(f"{results_mpmo_fuzzy}, Handoff: {results_mpmo_fuzzy['Handover']}")
        if impTech_hyst == True:
            print(f"{results_mpmo_fuzzy_hyst}, Handoff: {results_mpmo_fuzzy_hyst['Handover']}")
        if impTech_lockin == True:
            print(f"{results_mpmo_fuzzy_lockin}, Handoff: {results_mpmo_fuzzy_lockin['Handover']}")
        if impTech_TTT == True:
            print(f"{results_mpmo_fuzzy_ttt}, Handoff: {results_mpmo_fuzzy_ttt['Handover']}")
    if RMSE == True:
        print(f"{results_mpmo_rmse}, Handoff: {results_mpmo_rmse['Handover']}")
        if impTech_hyst == True:
            print(f"{results_mpmo_rmse_hyst}, Handoff: {results_mpmo_rmse_hyst['Handover']}")
        if impTech_lockin == True:
            print(f"{results_mpmo_rmse_lockin}, Handoff: {results_mpmo_rmse_lockin['Handover']}")
        if impTech_TTT == True:
            print(f"{results_mpmo_rmse_ttt}, Handoff: {results_mpmo_rmse_ttt['Handover']}")
    if TOPSIS_NN == True: 
        print(f"{results_nn_topsis}, Handoff: {results_nn_topsis['Handover']}")
    print(f"{results_benchmark}, Handoff: {results_benchmark['Handover']}")
    print("========================================================================================================================")
    
    final_results.append(results_list)
    indicators_results.append(copy.deepcopy(indicators_list))
    


def plot_results():
    global final_results, analyzed_parameters, weights_total_sum, indicators_results, iter_x_simu
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
    
    
    
    # ==================================================== Start - RMSE Analisys ====================================================
    print("========================================================================================================================")
    print("Indicators to Measure Deviation from a Benchmark")

    # Initialize a dictionary to store the summed values
    aggregated_results_rmse = {}
    
    # To use in 3D SSM
    SSE_3D_Graph = {}

    # Loop through each list in the indicators_results
    #print(f"Indicator Results: {indicators_results}")
    
    #print(f"Indicators Results: {indicators_results}")
    
    for group in indicators_results:
        for entry in group:
            algorithm = entry['Algorithm']

            if algorithm != 'Worst-Scenario':
                if algorithm not in aggregated_results_rmse:
                    aggregated_results_rmse[algorithm] = {param: 0 for param in entry if param != 'Algorithm'}
                
                for param in entry:
                    if param != 'Algorithm':
                        aggregated_results_rmse[algorithm][param] += sum([x**2 for x in entry[param]])

    
    SSE_3D_Graph = aggregated_results_rmse
    for agg in aggregated_results_rmse:
        for parm in aggregated_results_rmse[agg]:
            aggregated_results_rmse[agg][parm] = (aggregated_results_rmse[agg][parm] / iter_x_simu)**(1/2)
    
    
    print(f"Aggregated results: {aggregated_results_rmse}")
    
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
        k: [(v_i-min(v)) / (max(v)-min(v)) for v_i in v]
        for k, v in ind_dict.items()
    }
    print(normalized_rsme)
    
    simulations = indicators_results
    simulations_aux_max = []
    simulations_aux_min = []

    for sim in simulations:
        
        algo_aux_max = {}
        for att in analyzed_parameters:
            algo_aux_max[att] = []

        algo_aux_min = {}
        for att in analyzed_parameters:
            algo_aux_min[att] = []

        for algo in sim:
            for param, values_list in algo.items():
                if param != 'Algorithm':
                    
                    # Fill the algo_aux_max with zeros
                    if algo_aux_max[param] == []:
                        algo_aux_max[param] = [0] * len(values_list)    
                    i = 0
                    for value in values_list:
                        if value > algo_aux_max[param][i]:
                            algo_aux_max[param][i] = value
                        i = i + 1
                    
                    # Fill the algo_aux_min with 999999999999
                    if algo_aux_min[param] == []:
                        algo_aux_min[param] = [999999999999] * len(values_list)    
                    i = 0
                    for value in values_list:
                        if value < algo_aux_min[param][i]:
                            algo_aux_min[param][i] = value
                        i = i + 1
                    
        simulations_aux_max.append(algo_aux_max)
        simulations_aux_min.append(algo_aux_min)
    
    # Remove 'Worst-Scenario' entries
    cleaned_data = []
    for group in simulations:  # for each inner list
        new_group = [item for item in group if item['Algorithm'] != 'Worst-Scenario']
        cleaned_data.append(new_group)   
    simulations = cleaned_data
    

    i = 0
    for sim in simulations:
        j = 0
        for algo in sim:
            for param, values_list in algo.items():
                if param != 'Algorithm':
                    k = 0
                    for value in values_list:
                        if simulations_aux_max[i][param][k] == 0:
                            simulations[i][j][param][k] = 0
                        else:
                            simulations[i][j][param][k] = (simulations[i][j][param][k]) / (simulations_aux_max[i][param][k])
                        k = k + 1
            j = j + 1
        i = i + 1

    
    rmse_simulations = []
    for sim in simulations:
        rmse_sim = []
        for algo in sim:
            rmse_algo = [0] * len(simulations[0][0][next(iter(algo))])
            for i in range(len(simulations[0][0][next(iter(algo))])):
                j = 0
                for param, values_list in algo.items():
                    if param != 'Algorithm':
                        rmse_algo[i] = rmse_algo[i] + (weights[j]*(values_list[i]**2))
                    j = j + 1
            rmse_sim.append(rmse_algo)
        rmse_simulations.append(rmse_sim)

    rsme_final_results = []
    for rmse_simu in rmse_simulations:
        rsme_list_results = []
        for algo in rmse_simu:
            rmse_results = [(x / weights_total_sum)**(1/2) for x in algo]
            rsme_list_results.append(rmse_results)
            
        rsme_final_results.append(rsme_list_results)
        
    i = 0
    for sum_result_list in rsme_final_results:
        j = 0
        for sum_result in sum_result_list:
            rsme_final_results[i][j] = sum(rsme_final_results[i][j])
            j = j + 1
        i = i + 1

    results_sim_sum = [0] * len(rsme_final_results[0])
    for simu_sum in rsme_final_results:
        j = 0
        for sum_rsme in simu_sum:
            results_sim_sum[j] = results_sim_sum[j] + sum_rsme
            j = j + 1
    
    print("========================================================================================================================")
    print(results_sim_sum)

    x_min = min(results_sim_sum)
    x_max = max(results_sim_sum)
    rmse_per_index = [(x - x_min) / (x_max - x_min) for x in results_sim_sum]

    print("========================================================================================================================")
    print(f"RMSE of each Decision Maker: {rmse_per_index}")
    print("========================================================================================================================")
    print(f"Best option using RMSE: {rmse_per_index.index(min(rmse_per_index))}")
    print("========================================================================================================================")
    # ===================================================== End - RMSE Analisys =====================================================
    
    
    
    
    # ======================================================= Start - 3D Plot =======================================================
    if plots == True:
        ordered_algorithms = [algo for algo in SSE_3D_Graph]

        # Reorder parameters so 'Throughput' appears last
        parameters = list(next(iter(SSE_3D_Graph.values())).keys())
        parameters.remove("Throughput")
        parameters = parameters + ['Throughput']

        # Normalize each parameter individually
        param_norms = {}
        for param in parameters:
            values = [abs(SSE_3D_Graph[algo][param]) for algo in ordered_algorithms]
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
                raw_value = abs(SSE_3D_Graph[algo][param])
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
    # ======================================================= End - 3D Plot ======================================================= 
    
    
    
    
    
    # ================================================== Start - Bar Chart Plot ====================================================
    if plots == True:
        # Organize results by algorithm
        r_dict = {r['Algorithm']: r for r in results}
        print(f"{r_dict}")

        num_params = len(analyzed_parameters)

        # Split parameters into two halves
        half = math.ceil(num_params / 2)
        param_groups = [analyzed_parameters[:half], analyzed_parameters[half:]]

        # Loop through each group to create two separate figures
        for group_index, param_group in enumerate(param_groups):
            num_params_group = len(param_group)
            num_cols = 1
            num_rows = math.ceil(num_params_group / num_cols)
            
            # 1 - 3 columns
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(7, 1 * num_rows), constrained_layout=True)
            
            # 4 - 10 columns
            #fig, axes = plt.subplots(num_rows, num_cols, figsize=(7, 2 * num_rows), constrained_layout=True)
            
            # 11 - 17 columns
            #fig, axes = plt.subplots(num_rows, num_cols, figsize=(7, 3 * num_rows), constrained_layout=True)

            axes = axes.flatten() if num_params_group > 1 else [axes]

            for i, param in enumerate(param_group):
                values = [d[param] for d in r_dict.values()]
                labels = list(r_dict.keys())

                ax = axes[i]
                
                # 4 METHODS
                #hatches = ['', '', '', '', '']
                #bars = ax.bar(labels, values, color=["blue", "orange", "green", "red", "black"], edgecolor='black', linewidth=1.2)
                
                # SPMO + 4 METHODS
                #hatches = ['', '', '', '', '', '', '', '']
                #bars = ax.bar(labels, values, color=["grey", "grey", "grey", "blue", "orange", "green", "red", "black"], edgecolor='black', linewidth=1.2)
                
                # SPMO + 4 METHODS + IMPROVEMENT TECHENIQUES
                #hatches = ['', '', '', '', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '']
                #bars = ax.bar(labels, values, color=["silver", "silver", "silver", "gold", "gold", "gold", "gold", "blue", "blue", "blue", "blue", "green", "green", "green", "green", "red", "red", "red", "red", "purple", "purple", "purple", "purple", "green", "black"], edgecolor='black', linewidth=1.2)
                
                # 4 METHODS + RMSE + IMPROVEMENT TECHENIQUES
                #hatches = ['', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '|', '+', 'x', '', '']
                #bars = ax.bar(labels, values, color=["gold", "gold", "gold", "gold", "blue", "blue", "blue", "blue", "green", "green", "green", "green", "red", "red", "red", "red", "purple", "purple", "purple", "purple", "black"], edgecolor='black', linewidth=1.2)
                
                # 4 METHODS + RMSE
                #hatches = ['', '', '', '', '', '']
                #bars = ax.bar(labels, values, color=["blue", "orange", "green", "red", "purple", "black"], edgecolor='black', linewidth=1.2)
                
                # TOPSIS + TOPSIS NN
                hatches = ['', '']
                bars = ax.bar(labels, values, color=["blue", "orange", "black"], edgecolor='black', linewidth=1.2)
                
                # Apply hatch patterns to each bar
                for bar, hatch in zip(bars, hatches):
                    bar.set_hatch(hatch)
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

            #plt.savefig(f"outputs/bar_chart_part_{group_index + 1}.png", dpi=600, bbox_inches='tight')
            plt.show()
    # =================================================== End - Bar Chart Plot ====================================================



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
generate_random_WNS(predef_conf)

# Wireless Networks
connect_to_net(device_1)

# Optimization Methods
if SPMO_Methods == True:
    spmo_max_min_method_rssi = SPMO_MMM("SPMO-MAX-RSSI", "RSSI", True)
    spmo_max_min_method_snr = SPMO_MMM("SPMO-MAX-SNR", "SNR", True)
    spmo_pref = SPMO_Pref("SPMO-Preference", "Protocol", ['WiFi-5GHz', 'WiFi-2.4GHz', 'WiMax', 'LTE-4G', 'NB-IoT', 'LoRa-868'])
if SAW == True:
    mpmo_saw = MPMO_SAW("MPMO-SAW", analyzed_parameters, weights, directions)
    if impTech_hyst == True:
        mpmo_saw_hyst = MPMO_SAW("MPMO-SAW-Hysteresis", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
    if impTech_lockin == True:
        mpmo_saw_lockin = MPMO_SAW("MPMO-SAW-Lock", analyzed_parameters, weights, directions, lockin_percentage=lockin_percentage)
    if impTech_TTT == True:
        mpmo_saw_ttt = MPMO_SAW("MPMO-SAW-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
if WPM == True:
    mpmo_wpm = MPMO_WPM("MPMO-WPM", analyzed_parameters, weights, directions)
    if impTech_hyst == True:
        mpmo_wpm_hyst = MPMO_WPM("MPMO-WPM-Hyst", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
    if impTech_lockin == True:
        mpmo_wpm_lockin = MPMO_WPM("MPMO-WPM-Lock", analyzed_parameters, weights, directions, lockin_percentage=lockin_percentage)
    if impTech_TTT == True:
        mpmo_wpm_ttt = MPMO_WPM("MPMO-WPM-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
if TOPSIS == True:
    mpmo_topsis = MPMO_TOPSIS("MPMO-TOPSIS", analyzed_parameters, weights, directions)
    if impTech_hyst == True:
        mpmo_topsis_hyst = MPMO_TOPSIS("MPMO-TOPSIS-Hyst", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
    if impTech_lockin == True:
        mpmo_topsis_lockin = MPMO_TOPSIS("MPMO-TOPSIS-Lock", analyzed_parameters, weights, directions, lockin_percentage=lockin_percentage)
    if impTech_TTT == True:
        mpmo_topsis_ttt = MPMO_TOPSIS("MPMO-TOPSIS-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
if Fuzzy == True:
    mpmo_fuzzy = MPMO_Fuzzy("MPMO-Fuzzy")
    mpmo_fuzzy.definePresetConfigs()
    if impTech_hyst == True:
        mpmo_fuzzy_hyst = MPMO_Fuzzy("MPMO-Fuzzy-hyst", analyzed_parameters, directions, hysterese_percentage=hyst_percentage)
        mpmo_fuzzy_hyst.definePresetConfigs()
    if impTech_lockin == True:
        mpmo_fuzzy_lockin = MPMO_Fuzzy("MPMO-Fuzzy-Lock", analyzed_parameters, directions, lockin_percentage=lockin_percentage)
        mpmo_fuzzy_lockin.definePresetConfigs()
    if impTech_TTT == True:
        mpmo_fuzzy_ttt = MPMO_Fuzzy("MPMO-Fuzzy-TimeToTrigger", time_to_trigger=tt_trigger)
        mpmo_fuzzy_ttt.definePresetConfigs()
if RMSE == True:
    mpmo_rmse = MPMO_RMSE("MPMO-RMSE", analyzed_parameters, weights, directions)
    if impTech_hyst == True:
        mpmo_rmse_hyst = MPMO_RMSE("MPMO-RMSE-Hyst", analyzed_parameters, weights, directions, hysterese_percentage=hyst_percentage)
    if impTech_lockin == True:
        mpmo_rmse_lockin = MPMO_RMSE("MPMO-RMSE-LockIn", analyzed_parameters, weights, directions, lockin_percentage=lockin_percentage)
    if impTech_TTT == True:
        mpmo_rmse_ttt = MPMO_RMSE("MPMO-RMSE-TimeToTrigger", analyzed_parameters, weights, directions, time_to_trigger=tt_trigger)
if TOPSIS_NN == True: 
    nn_topsis = NN_TOPSIS("NN-TOPSIS", analyzed_parameters)
benchmark = BenchmarkMethod("Benchmark", analyzed_parameters, directions)
worst_scenario = WorstScenarioMethod("Worst-Scenario", analyzed_parameters, directions)

# Instances of Performance Analysis
if SPMO_Methods == True:
    p_spmo_max_min_rssi = PerformanceAnalysis("MAX-RSSI")
    p_spmo_max_min_snr = PerformanceAnalysis("MAX-SNR")
    p_spmo_pref = PerformanceAnalysis("Preference")
if SAW == True:
    p_mpmo_saw = PerformanceAnalysis("SAW")
    if impTech_hyst == True:
        p_mpmo_saw_hyst = PerformanceAnalysis("SAW-H")
    if impTech_lockin == True:
        p_mpmo_saw_lockin = PerformanceAnalysis("SAW-Lock")
    if impTech_TTT == True:
        p_mpmo_saw_ttt = PerformanceAnalysis("SAW-TTT")
if WPM == True:
    p_mpmo_wpm = PerformanceAnalysis("WPM")
    if impTech_hyst == True:
        p_mpmo_wpm_hyst = PerformanceAnalysis("WPM-H")
    if impTech_lockin == True:
        p_mpmo_wpm_lockin = PerformanceAnalysis("WPM-Lock")
    if impTech_TTT == True:
        p_mpmo_wpm_ttt = PerformanceAnalysis("WPM-TTT")
if TOPSIS == True:
    p_mpmo_topsis = PerformanceAnalysis("TOPSIS")
    if impTech_hyst == True:
        p_mpmo_topsis_hyst = PerformanceAnalysis("TOPSIS-H")
    if impTech_lockin == True:
        p_mpmo_topsis_lockin = PerformanceAnalysis("TOPSIS-Lock")
    if impTech_TTT == True:
        p_mpmo_topsis_ttt = PerformanceAnalysis("TOPSIS-TTT")
if Fuzzy == True:
    p_mpmo_fuzzy = PerformanceAnalysis("Fuzzy")
    if impTech_hyst == True:
        p_mpmo_fuzzy_hyst = PerformanceAnalysis("Fuzzy-H")
    if impTech_lockin == True:
        p_mpmo_fuzzy_lockin = PerformanceAnalysis("Fuzzy-Lock")
    if impTech_TTT == True:
        p_mpmo_fuzzy_ttt = PerformanceAnalysis("Fuzzy-TTT")
if RMSE == True:
    p_mpmo_rmse = PerformanceAnalysis("RMSE")
    if impTech_hyst == True:
        p_mpmo_rmse_hyst = PerformanceAnalysis("RMSE-H")
    if impTech_lockin == True:
        p_mpmo_rmse_lockin = PerformanceAnalysis("RMSE-Lock")
    if impTech_TTT == True:
        p_mpmo_rmse_ttt = PerformanceAnalysis("RMSE-TTT")
if TOPSIS_NN == True: 
    p_nn_topsis = PerformanceAnalysis("NN-TOPSIS")
p_benchmark = PerformanceAnalysis("Benchmark")
p_worst_scenario = PerformanceAnalysis("Worst-Scenario")

update_position(device_1)
root.mainloop()