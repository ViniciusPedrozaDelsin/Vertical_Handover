import tkinter as tk
from tkinter import ttk
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


# Wireless Network Systems
WNS_list = []

# WiFi's
wifi_1 = MNS("WiFi-1", 400, 60, 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_1)

wifi_2 = MNS("WiFi-2", 300, 30, 20, 5000000000, 80000000, 15, "WiFi-5GHz", maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_2)

wifi_3 = MNS("WiFi-3", 100, 300, 20, 5000000000, 80000000, 15, "WiFi-5GHz", maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_3)

wifi_4 = MNS("WiFi-4", 800, 800, 20, 5000000000, 80000000, 15, "WiFi-5GHz", maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_4)

wifi_5 = MNS("WiFi-5", 150, 750, 20, 5000000000, 80000000, 15, "WiFi-5GHz", maximum_radius=90, predef_throughput=[1000000000, 500000000, 150000000], predef_snr=[40, 25, 15], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_5)

wifi_6 = MNS("WiFi-6", 900, 500, 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_6)

wifi_7 = MNS("WiFi-7", 480, 620, 20, 2400000000, 20000000, 10, "WiFi-2.4GHz", maximum_radius=150, predef_throughput=[200000000, 100000000, 30000000], predef_snr=[40, 25, 10], predef_rssi=[-50, -65, -80])
WNS_list.append(wifi_7)


# NB-IoT 5G
nbiot_5g_1 = MNS("NBIoT-5g-1", -6500, -6500, 30, 800000000, 1400000, 2, "NB-IoT-5G", maximum_radius=15000, predef_throughput=[100000, 50000, 10000], predef_snr=[10, 5, 2], predef_rssi=[-90, -105, -115])
WNS_list.append(nbiot_5g_1)


# LoRa's
LoRa_1 = MNS("LoRa-1", 7100, -7250, 14, 868000000, 250000, 0, "LoRa-868", maximum_radius=10000, predef_throughput=[50000, 10000, 1000], predef_snr=[10, 5, 0], predef_rssi=[-80, -100, -120])
WNS_list.append(LoRa_1)

LoRa_2 = MNS("LoRa-2", 7100, -7250, 14, 868000000, 250000, 0, "LoRa-868", maximum_radius=10000, predef_throughput=[50000, 10000, 1000], predef_snr=[10, 5, 0], predef_rssi=[-80, -100, -120])
WNS_list.append(LoRa_2)


# LTE 4G
LTE_4g = MNS("LTE-4g-1", 13000, 13000, 40, 868000000, 250000, 5, "LTE-4G", maximum_radius=30000, predef_throughput=[100000000, 40000000, 5000000], predef_snr=[15, 10, 5], predef_rssi=[-70, -90, -100])
WNS_list.append(LTE_4g)


# WiFi Max
wifi_max_1 = MNS("WiFi-Max-1", -2000, 3900, 40, 3000000000, 10000000, 10, "WiFi-Max", maximum_radius=6000, predef_throughput=[40000000, 15000000, 2000000], predef_snr=[15, 10, 5], predef_rssi=[-60, -80, -90])
WNS_list.append(wifi_max_1)

'''
print(wifi_1.transmission_range())
device_1 = Device(1, 139.6, 0)
device_1.connect_to_network(wifi_1)
print(device_1.get_QoS_Parameters(wifi_1))

wifi_2 = MNS("WiFi-2", 0, 0, 20, 5000000000, 80000000, 100, True)

# typically uses QPSK or 16-QAM
# Spectral efficiency for QPSK is around 0.5 bps/Hz.
# Spectral efficiency for 16-QAM is around 1 bps/Hz.
# In real-world conditions, the efficiency factor can be around 0.2 - 0.5, depending on the environment and interference.
nbiot_5g_1 = MNS("NBIoT-5g-1", 0, 0, 50, 800000000, 1400000, 100, True)'''

'''
device_1 = Device(1, 10, 20)
device_1.connect_to_network(wifi_1)
print(device_1.get_QoS_Parameters(wifi_1))

device_1.connect_to_network(wifi_2)
print(device_1.get_QoS_Parameters(wifi_2))

device_1.connect_to_network(nbiot_5g_1)
print(device_1.get_QoS_Parameters(nbiot_5g_1))
'''


# =============================== Initialize Graph ===============================
# Initialize parameters
x_max, y_max = 1000, 1000
x, y = 500, 500  # Start position in the center

def random_direction():
    angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
    dx = 15 * np.cos(angle)
    dy = 15 * np.sin(angle)
    return dx, dy

def update_position(device):
    global x, y
    dx, dy = random_direction()
    x = min(max(x + dx, 0), x_max)
    y = min(max(y + dy, 0), y_max)
    calculate_parameters(device, round(x,4), round(y,4))
    plot_graph()
    # Call again after 1 second
    root.after(100, lambda: update_position(device))

def calculate_parameters(device, x_position, y_position):
    print(f"x:{x_position} || y:{y_position}")
    device.updatePosition(x_position, y_position)
    print("===================================================")
    #print(f"Networks: {device.get_all_QoS_Parameters_predef()}")
    device.get_all_QoS_Parameters_predef()
    available_networks = device.get_available_networks()
    print(f"Available Networks: {available_networks}")
    decision_spmo_mmm = device_1.makeDecision(spmo_max_min_method, available_networks)
    print(f"Decision SPMO MAX MIN: {decision_spmo_mmm}")
    decision_spmo_pref = device_1.makeDecision(spmo_pref, available_networks)
    print(f"Decision SPMO Preference: {decision_spmo_pref}")
    decision_mpmo_saw = device_1.makeDecision(mpmo_saw, available_networks)
    print(f"Decision MPMO SAW: {decision_mpmo_saw}")
    decision_mpmo_wpm = device_1.makeDecision(mpmo_wpm, available_networks)
    print(f"Decision MPMO WPM: {decision_mpmo_wpm}")
    print("===================================================")
    

def plot_graph():
    ax.clear()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Wifi 1, 2.4GHz
    circle = Circle((400, 60), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((400, 60), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((400, 60), 50, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 2, 5GHz
    circle = Circle((300, 30), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((300, 30), 60, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((300, 30), 30, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 3, 5GHz
    circle = Circle((100, 300), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((100, 300), 60, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((100, 300), 30, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 4, 5GHz
    circle = Circle((800, 800), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((800, 800), 60, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((800, 800), 30, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 5, 5GHz
    circle = Circle((150, 750), 90, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((150, 750), 60, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((150, 750), 30, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 6, 2.4GHz
    circle = Circle((900, 500), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((900, 500), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((900, 500), 50, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # Wifi 7, 2.4GHz
    circle = Circle((480, 620), 150, color='green', alpha=0.15, fill=True)
    ax.add_patch(circle)
    circle = Circle((480, 620), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    circle = Circle((480, 620), 50, color='green', alpha=0.6, fill=True)
    ax.add_patch(circle)
    
    # NB-IoT 5g
    circle = Circle((-6500, -6500), 15000, color='blue', alpha=0.05, fill=True)
    ax.add_patch(circle)
    circle = Circle((-6500, -6500), 10000, color='blue', alpha=0.1, fill=True)
    ax.add_patch(circle)
    circle = Circle((-6500, -6500), 5000, color='blue', alpha=0.2, fill=True)
    ax.add_patch(circle)
    
    # LoRa 1
    circle = Circle((7100, -7250), 10000, color='yellow', alpha=0.05, fill=True)
    ax.add_patch(circle)
    circle = Circle((7100, -7250), 6666, color='yellow', alpha=0.1, fill=True)
    ax.add_patch(circle)
    circle = Circle((7100, -7250), 3333, color='yellow', alpha=0.2, fill=True)
    ax.add_patch(circle)
    
    # LoRa 2
    circle = Circle((500, 7000), 10000, color='yellow', alpha=0.05, fill=True)
    ax.add_patch(circle)
    circle = Circle((500, 7000), 6666, color='yellow', alpha=0.1, fill=True)
    ax.add_patch(circle)
    circle = Circle((500, 7000), 3333, color='yellow', alpha=0.2, fill=True)
    ax.add_patch(circle)
    
    # LTE 4g 1
    circle = Circle((13000, 13000), 30000, color='red', alpha=0.04, fill=True)
    ax.add_patch(circle)
    circle = Circle((13000, 13000), 20000, color='red', alpha=0.04, fill=True)
    ax.add_patch(circle)
    circle = Circle((13000, 13000), 10000, color='red', alpha=0.04, fill=True)
    ax.add_patch(circle)
    
    # WiFi Max 1
    circle = Circle((-2000, 3900), 6000, color='purple', alpha=0.02, fill=True)
    ax.add_patch(circle)
    circle = Circle((-2000, 3900), 4000, color='purple', alpha=0.1, fill=True)
    ax.add_patch(circle)
    circle = Circle((-2000, 3900), 2000, color='purple', alpha=0.1, fill=True)
    ax.add_patch(circle)
    
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
spmo_max_min_method = SPMO_MMM("SPMO-MAX-RSSI", "RSSI", True)
spmo_pref = SPMO_Pref("SPMO-Preference", "Protocol", ['WiFi-5GHz', 'WiFi-2.4GHz', 'NB-IoT-5G', 'LoRa-868', 'WiFi-Max', 'LTE-4G'])
mpmo_saw = MPMO_SAW("MPMO-SAW", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])
mpmo_wpm = MPMO_WPM("MPMO-WPM", ["RSSI", "SNR", "Throughput", "Distance"], [3, 5, 10, 2], [1, 1, 1, 0])

update_position(device_1)
root.mainloop()