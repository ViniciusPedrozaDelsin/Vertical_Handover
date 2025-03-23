import tkinter as tk
from tkinter import ttk
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from WirelessNetworkSystem import WirelessNetworkSystem as MNS
from Device import Device

wifi_1 = MNS("WiFi-1", 0, 0, 20, 2400000000, 20000000, 15, True)

wifi_2 = MNS("WiFi-2", 0, 0, 20, 5000000000, 80000000, 100, True)

# typically uses QPSK or 16-QAM
# Spectral efficiency for QPSK is around 0.5 bps/Hz.
# Spectral efficiency for 16-QAM is around 1 bps/Hz.
# In real-world conditions, the efficiency factor can be around 0.2 - 0.5, depending on the environment and interference.
nbiot_5g_1 = MNS("NBIoT-5g-1", 0, 0, 50, 800000000, 1400000, 100, True)
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
x_max, y_max = 500, 500
x, y = 250, 250  # Start position in the center

def random_direction():
    angle = random.uniform(0, 2 * np.pi)  # Random angle in radians
    dx = 10 * np.cos(angle)
    dy = 10 * np.sin(angle)
    return dx, dy

def update_position(device, wifi_1):
    global x, y
    dx, dy = random_direction()
    x = min(max(x + dx, 0), x_max)
    y = min(max(y + dy, 0), y_max)
    #calculate_parameters(device, wifi_1, round(x,4), round(y,4))
    print(f"x:{round(x,4)} || y:{round(y,4)}")
    device.updatePosition(round(x,4), round(y,4))
    print(device.get_QoS_Parameters(wifi_1))
    plot_graph()
    root.after(1000, lambda: update_position(device, wifi_1))  # Call again after 1 second

def calculate_parameters(device, wifi_1, x_position, y_position):
    pass
    

def plot_graph():
    ax.clear()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    # Wifi 1, 2.4GHz
    circle = Circle((0, 0), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    '''
    # Wifi 1, 2.4GHz
    circle = Circle((450, 60), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Wifi 2, 2.4GHz
    circle = Circle((80, 300), 90, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Wifi 3, 5GHz
    circle = Circle((380, 290), 80, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Wifi 4, 5GHz
    circle = Circle((290, 360), 70, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Wifi 5, 2.4GHz
    circle = Circle((170, 170), 100, color='green', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Radio base 1
    circle = Circle((-450, -800), 1200, color='blue', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # Radio base 2
    circle = Circle((1000, -750), 1200, color='blue', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # LoRa 1
    circle = Circle((800, 900), 800, color='red', alpha=0.3, fill=True)
    ax.add_patch(circle)
    
    # LoRa 2
    circle = Circle((-370, 250), 600, color='red', alpha=0.3, fill=True)
    ax.add_patch(circle)'''
    
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
device_1.connect_to_network(wifi_1)
print(device_1.get_QoS_Parameters(wifi_1))

update_position(device_1, wifi_1)
root.mainloop()