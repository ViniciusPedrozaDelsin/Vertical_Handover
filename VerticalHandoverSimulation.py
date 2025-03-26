import tkinter as tk
from tkinter import ttk
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from WirelessNetworkSystem import WirelessNetworkSystem as MNS
from Device import Device
from SPMO_Max_Min_Method import SPMO_Max_Min_Method as SPMO_MMM
from SPMO_Preference import SPMO_Preference as SPMO_Pref
from MPMO_SAW import MPMO_SAW
from MPMO_WPM import MPMO_WPM
from MPMO_TOPSIS import MPMO_TOPSIS
from PerformanceAnalysis import PerformanceAnalysis



# ==================================== Initial Parameters ====================================
# Map dimension
x_max, y_max = 1000, 1000

# Start position
x, y = random.uniform(0, x_max), random.uniform(0, y_max)

# Interval between iterations
iter_interval = 10

# Distance for iteration
dist_iter = 15

# n = Number of iterations, j = DO NOT CHANGE
j = 0
n = 1000
# ============================================================================================



# Wireless Network Systems
WNS_list = []

# WiFi's
wifi_1 = MNS("WiFi-1", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_1, 'green', 0.3])

wifi_2 = MNS("WiFi-2", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_2, 'green', 0.3])

wifi_3 = MNS("WiFi-3", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_3, 'green', 0.3])

wifi_4 = MNS("WiFi-4", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_4, 'green', 0.3])

wifi_5 = MNS("WiFi-5", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_5, 'green', 0.3])

wifi_6 = MNS("WiFi-6", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_6, 'green', 0.3])

wifi_7 = MNS("WiFi-7", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_7, 'green', 0.3])

wifi_8 = MNS("WiFi-8", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_8, 'green', 0.3])

wifi_9 = MNS("WiFi-9", random.uniform(0, x_max), random.uniform(0, y_max), 20, 5000000000, 80000000, 15, "WiFi-5GHz", 0.50, 1, maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_9, 'green', 0.3])

wifi_10 = MNS("WiFi-10", random.uniform(0, x_max), random.uniform(0, y_max), 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", 0.50, 1, maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append([wifi_10, 'green', 0.3])


# NB-IoT 5G
nbiot_5g_1 = MNS("NBIoT-5g-1", random.uniform(-10*x_max, 10*x_max), random.uniform(-10*y_max, 10*y_max), 30, 800000000, 1400000, 2, "NB-IoT-5G", 0.25, 5, maximum_radius=15000, predef_throughput=[100000, 50000, 10000], predef_snr=[10, 5, 2], predef_rssi=[-90, -105, -115])
WNS_list.append([nbiot_5g_1, 'blue', 0.03])


# LoRa's
LoRa_1 = MNS("LoRa-1", random.uniform(-7*x_max, 7*x_max), random.uniform(-7*y_max, 7*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 10000, 1000], predef_snr=[10, 5, 0], predef_rssi=[-80, -100, -120])
WNS_list.append([LoRa_1, 'yellow', 0.03])

LoRa_2 = MNS("LoRa-2", random.uniform(-7*x_max, 7*x_max), random.uniform(-7*y_max, 7*y_max), 14, 868000000, 250000, 0, "LoRa-868", 0.05, 1, maximum_radius=10000, predef_throughput=[50000, 10000, 1000], predef_snr=[10, 5, 0], predef_rssi=[-80, -100, -120])
WNS_list.append([LoRa_2, 'yellow', 0.03])


# LTE 4G
LTE_4g = MNS("LTE-4g-1", random.uniform(-20*x_max, 20*x_max), random.uniform(-20*y_max, 20*y_max), 40, 868000000, 250000, 5, "LTE-4G", 1.05, 3, maximum_radius=30000, predef_throughput=[100000000, 40000000, 5000000], predef_snr=[15, 10, 5], predef_rssi=[-70, -90, -100])
WNS_list.append([LTE_4g, 'red', 0.03])


# WiFi Max
wifi_max_1 = MNS("WiFi-Max-1", random.uniform(-4*x_max, 4*x_max), random.uniform(-4*y_max, 4*y_max), 40, 3000000000, 10000000, 10, "WiFi-Max", 0.80, 2, maximum_radius=6000, predef_throughput=[40000000, 15000000, 2000000], predef_snr=[15, 10, 5], predef_rssi=[-60, -80, -90])
WNS_list.append([wifi_max_1, 'purple', 0.03])





# =============================== Initialize Graph ===============================

def random_direction():
    global dist_iter
    angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
    dx = dist_iter * np.cos(angle)
    dy = dist_iter * np.sin(angle)
    return dx, dy

def update_position(device):
    global x, y, j, n, iter_interval
    dx, dy = random_direction()
    x = min(max(x + dx, 0), x_max)
    y = min(max(y + dy, 0), y_max)
    calculate_parameters(device, round(x,4), round(y,4))
    plot_graph()
    # Call again after 1 second
    j += 1
    if j < n:
        root.after(iter_interval, lambda: update_position(device))
    else:
        performe_analysis()
        sys.exit()

def calculate_parameters(device, x_position, y_position):
    print(f"x:{x_position} || y:{y_position}")
    device.updatePosition(x_position, y_position)
    print("===================================================")
    #print(f"Networks: {device.get_all_QoS_Parameters_predef()}")
    device.get_all_QoS_Parameters_predef()
    available_networks = device.get_available_networks()
    print(f"Available Networks: {available_networks}")
    
    decision_spmo_mmm_rssi = device_1.makeDecision(spmo_max_min_method_rssi, available_networks)
    p_spmo_max_min_rssi.store_QoS_parameters(decision_spmo_mmm_rssi)
    print(f"Decision SPMO MAX MIN RSSI: {decision_spmo_mmm_rssi}")
    
    decision_spmo_mmm_snr = device_1.makeDecision(spmo_max_min_method_snr, available_networks)
    p_spmo_max_min_snr.store_QoS_parameters(decision_spmo_mmm_snr)
    print(f"Decision SPMO MAX MIN SNR: {decision_spmo_mmm_snr}")
    
    decision_spmo_pref = device_1.makeDecision(spmo_pref, available_networks)
    p_spmo_pref.store_QoS_parameters(decision_spmo_pref)
    print(f"Decision SPMO Preference: {decision_spmo_pref}")
    
    decision_mpmo_saw = device_1.makeDecision(mpmo_saw, available_networks)
    p_mpmo_saw.store_QoS_parameters(decision_mpmo_saw)
    print(f"Decision MPMO SAW: {decision_mpmo_saw}")
    
    decision_mpmo_wpm = device_1.makeDecision(mpmo_wpm, available_networks)
    p_mpmo_wpm.store_QoS_parameters(decision_mpmo_wpm)
    print(f"Decision MPMO WPM: {decision_mpmo_wpm}")
    
    decision_mpmo_topsis = device_1.makeDecision(mpmo_topsis, available_networks)
    p_mpmo_topsis.store_QoS_parameters(decision_mpmo_topsis)
    print(f"Decision MPMO TOPSIS: {decision_mpmo_topsis}")
    
    print("===================================================")

def performe_analysis():
    # Performance Analysis
    print(f"{p_spmo_max_min_rssi.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_spmo_max_min_rssi.count_number_of_handovers()}")
    print(f"{p_spmo_max_min_snr.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_spmo_max_min_snr.count_number_of_handovers()}")
    print(f"{p_spmo_pref.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_spmo_pref.count_number_of_handovers()}")
    print(f"{p_mpmo_saw.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_mpmo_saw.count_number_of_handovers()}")
    print(f"{p_mpmo_wpm.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_mpmo_wpm.count_number_of_handovers()}")
    print(f"{p_mpmo_topsis.calculate_average_QoS_parameters(['RSSI', 'SNR', 'Throughput', 'PC', 'MC'])}, Handoff: {p_mpmo_topsis.count_number_of_handovers()}")

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
    
    
    '''
    # Wifi 1, 2.4GHz
    circle = Circle((400, 60), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((400, 60), 100, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((400, 60), 50, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 2, 5GHz
    circle = Circle((300, 30), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((300, 30), 60, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((300, 30), 30, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 3, 5GHz
    circle = Circle((100, 300), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((100, 300), 60, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((100, 300), 30, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 4, 5GHz
    circle = Circle((800, 800), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((800, 800), 60, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((800, 800), 30, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 5, 5GHz
    circle = Circle((150, 750), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((150, 750), 60, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((150, 750), 30, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 6, 2.4GHz
    circle = Circle((900, 500), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((900, 500), 100, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((900, 500), 50, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # Wifi 7, 2.4GHz
    circle = Circle((480, 620), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    # circle = Circle((480, 620), 100, color='green', alpha=0.3, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((480, 620), 50, color='green', alpha=0.6, fill=True)
    # ax.add_patch(circle)
    
    # NB-IoT 5g
    circle = Circle((-6500, -6500), 15000, color='blue', alpha=0.05, fill=True)
    ax.add_patch(circle)
    # circle = Circle((-6500, -6500), 10000, color='blue', alpha=0.1, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((-6500, -6500), 5000, color='blue', alpha=0.2, fill=True)
    # ax.add_patch(circle)
    
    # LoRa 1
    circle = Circle((7100, -7250), 10000, color='yellow', alpha=0.05, fill=True)
    ax.add_patch(circle)
    # circle = Circle((7100, -7250), 6666, color='yellow', alpha=0.05, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((7100, -7250), 3333, color='yellow', alpha=0.05, fill=True)
    # ax.add_patch(circle)
    
    # LoRa 2
    circle = Circle((500, 3500), 10000, color='yellow', alpha=0.05, fill=True)
    ax.add_patch(circle)
    # circle = Circle((500, 3500), 6666, color='yellow', alpha=0.05, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((500, 3500), 3333, color='yellow', alpha=0.05, fill=True)
    # ax.add_patch(circle)
    
    # LTE 4g 1
    circle = Circle((13000, 13000), 30000, color='red', alpha=0.04, fill=True)
    ax.add_patch(circle)
    # circle = Circle((13000, 13000), 20000, color='red', alpha=0.04, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((13000, 13000), 10000, color='red', alpha=0.04, fill=True)
    # ax.add_patch(circle)
    
    # WiFi Max 1
    circle = Circle((-2000, 3900), 6000, color='purple', alpha=0.02, fill=True)
    ax.add_patch(circle)
    # circle = Circle((-2000, 3900), 4000, color='purple', alpha=0.1, fill=True)
    # ax.add_patch(circle)
    # circle = Circle((-2000, 3900), 2000, color='purple', alpha=0.1, fill=True)
    # ax.add_patch(circle)
    '''
    ax.scatter(x, y, color='red', s=50)
    ax.set_title("Random Walk Simulation")
    canvas.draw()


# Create GUI
root = tk.Tk()
root.title("Random Walk Visualization")

frame = ttk.Frame(root)
frame.pack()

fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack()

device_1 = Device(1, x, y)

# Wireless Networks
device_1.connect_to_network(wifi_1)
device_1.connect_to_network(wifi_2)
device_1.connect_to_network(wifi_3)
device_1.connect_to_network(wifi_4)
device_1.connect_to_network(wifi_5)
device_1.connect_to_network(wifi_6)
device_1.connect_to_network(wifi_7)
device_1.connect_to_network(nbiot_5g_1)
device_1.connect_to_network(LoRa_1)
device_1.connect_to_network(LoRa_2)
device_1.connect_to_network(LTE_4g)
device_1.connect_to_network(wifi_max_1)

# Optimization Methods
spmo_max_min_method_rssi = SPMO_MMM("SPMO-MAX-RSSI", "RSSI", True)
spmo_max_min_method_snr = SPMO_MMM("SPMO-MAX-SNR", "SNR", True)
spmo_pref = SPMO_Pref("SPMO-Preference", "Protocol", ['WiFi-5GHz', 'WiFi-2.4GHz', 'NB-IoT-5G', 'LoRa-868', 'WiFi-Max', 'LTE-4G'])
mpmo_saw = MPMO_SAW("MPMO-SAW", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
mpmo_wpm = MPMO_WPM("MPMO-WPM", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
mpmo_topsis = MPMO_TOPSIS("MPMO-TOPSIS", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])

# Instances of Performance Analysis
p_spmo_max_min_rssi = PerformanceAnalysis("SPMO-MAX-RSSI")
p_spmo_max_min_snr = PerformanceAnalysis("SPMO-MAX-SNR")
p_spmo_pref = PerformanceAnalysis("SPMO-Preference")
p_mpmo_saw = PerformanceAnalysis("MPMO-SAW")
p_mpmo_wpm = PerformanceAnalysis("MPMO-WPM")
p_mpmo_topsis = PerformanceAnalysis("MPMO_TOPSIS")

update_position(device_1)
root.mainloop()
