import numpy as np
from scipy.stats import rice
import math
import random

class WirelessNetworkSystem:
    def __init__(self, system_name, x_position, y_position, transmission_power_dbm, frequency, bandwidth, minimum_snr, protocol, power_consumption, monetary_cost, maximum_radius=None, predef_throughput=None, predef_snr=None, predef_rssi=None, predef_ber=None, predef_fec=None, predef_config=True, corrections_real_world_applications=False, fading=None):
        self.system_name = system_name
        self.connected_devices = set()
        
        # System Configs
        self.x_position = x_position
        self.y_position = y_position
        self.transmission_power_dbm = transmission_power_dbm
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.minimum_snr = minimum_snr
        self.protocol = protocol
        self.power_consumption = power_consumption
        self.monetary_cost = monetary_cost
        self.corrections_real_world_applications = corrections_real_world_applications
        self.predef_config = predef_config
        self.fading = fading
        if self.predef_config == True:
            self.maximum_radius = maximum_radius
        else:
            self.maximum_radius = self.transmission_range()
        
        # Pre-defined Configs
        self.predef_throughput = predef_throughput
        self.predef_snr = predef_snr
        self.predef_rssi = predef_rssi
        self.predef_ber = predef_ber
        self.predef_fec = predef_fec
        
    
    def attach_device(self, device):
        # Attach a device to this network
        self.connected_devices.add(device)
        device.networks.add(self)
        
    def detach_device(self, device):
        # Detach a device from this network.
        if device in self.connected_devices:
            self.connected_devices.remove(device)
            device.networks.remove(self)
    
    def transmission_range(self):
        # Boltzmann's Constant (J/K)
        k = 1.38 * (10**-23)
        # Temperature 290 Kelvin
        T = 290
        # Speed of Light
        c = 300000000
        if self.corrections_real_world_applications == True:
            transmission_range_radius = 10 ** ((self.transmission_power_dbm/20) - (math.log10(self.frequency)) - (math.log10((4*math.pi)/c)) - (0.5*(math.log10(k*T*self.bandwidth))) - (1.5) - (self.minimum_snr/20) - (1) - (0.5*math.log10(2)))
        else:
            transmission_range_radius = 10 ** ((self.transmission_power_dbm/20) - (math.log10(self.frequency)) - (math.log10((4*math.pi)/c)) - (0.5*(math.log10(k*T*self.bandwidth))) - (1.5) - (self.minimum_snr/20))
        return transmission_range_radius
    
    def update_transmission_range(self, fading):
        # Boltzmann's Constant (J/K)
        k = 1.38 * (10**-23)
        # Temperature 290 Kelvin
        T = 290
        # Speed of Light
        c = 300000000
        if self.corrections_real_world_applications == True:
            transmission_range_radius = 10 ** ((self.transmission_power_dbm/20) - (math.log10(self.frequency)) - (math.log10((4*math.pi)/c)) - (0.5*(math.log10(k*T*self.bandwidth))) - (1.5) - (self.minimum_snr/20) - (1) - (0.5*math.log10(2)) + (fading/20))
        else:
            transmission_range_radius = 10 ** ((self.transmission_power_dbm/20) - (math.log10(self.frequency)) - (math.log10((4*math.pi)/300000000)) - (0.5*(math.log10(k*290*self.bandwidth))) - (1.5) - (self.minimum_snr/20) + (fading/20))
            
        self.maximum_radius = transmission_range_radius
    
    def calculateDeviceDistance(self, device):
        distance = (((device.x_position - self.x_position)**2) + ((device.y_position - self.y_position))**2)**(1/2)
        return distance
    
    # Free Space Path Loss
    def calculateFSPL_db(self, d):
        # Distance correction if d = 0
        if d == 0: d = d + 0.00001
        # Speed of Light
        c = 300000000
        if self.corrections_real_world_applications == True:
            # Add Arbitrary Loss = 20db
            free_space_path_loss_db = (20*math.log10(d)) + (20*math.log10(self.frequency)) + (20*math.log10((4*math.pi)/c)) + 20
            # FSPL correction if it is less then twenty
            if free_space_path_loss_db < 20: free_space_path_loss_db = 20
        else:
            # FSPL "Vanilla"
            free_space_path_loss_db = (20*math.log10(d)) + (20*math.log10(self.frequency)) + (20*math.log10((4*math.pi)/c))
            # FSPL correction if it is less then zero
            if free_space_path_loss_db < 0: free_space_path_loss_db = 0
        return free_space_path_loss_db
    
    # Rayleigh Fading (for NLOS, no dominant path)
    def calculateRayleighFading_db(self):
        rayleigh_fading = np.random.rayleigh(scale=1, size=1000)
        fading_db = 20 * np.log10(rayleigh_fading)
        mean_rayleigh_samples = sum(fading_db) / len(fading_db)
        return mean_rayleigh_samples
    
    # Rician Fading (for LOS + multipath)
    def calculateRicianFading_db(self, K_dB, sigma=1, num_samples=1000):
        K = 10 ** (K_dB / 10)       # Convert K-factor to linear scale
        nu = np.sqrt(2 * K * sigma**2)  # LOS component amplitude
        rician_samples = rice.rvs(b=nu / sigma, scale=sigma, size=num_samples)
        mean_rician_samples = sum(rician_samples) / len(rician_samples)
        return mean_rician_samples
    
    # Calculate Shadowing
    def calculateShadowing(self, sigma=0.5):
        shadowing = np.random.normal(loc=0.0, scale=sigma)
        return shadowing

    def calculateCOST231HataModel(self, d):
        pass
        
    def calculateRSSI_dbm(self, fspl):
        rssi_dbm = self.transmission_power_dbm - fspl
        return rssi_dbm
        
    def calculateThermalNoise_dbm(self, NF=0):
        # Boltzmann's Constant (J/K)
        k = 1.38 * (10**-23)
        # Temperature 290 Kelvin
        T = 290
        # Thermal Noise Formula
        thermal_noise_power_dbm = 10 * math.log10(k*T*self.bandwidth) + 30 + NF
        return thermal_noise_power_dbm
       
    def calculateSNR_db(self, rssi_dbm, thermal_noise_power_dbm):
        snr = (10**(rssi_dbm/10)) / (10**(thermal_noise_power_dbm/10))
        if self.corrections_real_world_applications == True:
            # Correction of 0.7 in the SNR for real world applications
            snr_db = 10 * math.log10(snr*0.5)
        else:
            snr_db = 10 * math.log10(snr)
        #return round(snr_db, 3)
        return snr_db
    
    def calculateMinimumSNR_db(self, snr_db):
        if snr_db >= self.minimum_snr:
            status = "Online"
        else:
            status = "Offline"
        return status
    
    def calculateChannelCapacity(self, snr_db):
        channel_capacity = self.bandwidth * math.log2(1 + (10**(snr_db/10)))
        return channel_capacity
    
    def estimateThroughput(self, snr_db, cc):
        if snr_db >= 30:
            estimated_throughput = cc * 0.8
        elif snr_db >= 25 and snr_db < 30:
            estimated_throughput = cc * 0.7
        elif snr_db >= 20 and snr_db < 25:
            estimated_throughput = cc * 0.5
        elif snr_db >= 15 and snr_db < 20:
            estimated_throughput = cc * 0.4
        elif snr_db >= 10 and snr_db < 15:
            estimated_throughput = cc * 0.3
        elif snr_db >= 5 and snr_db < 10:
            estimated_throughput = cc * 0.2
        elif snr_db >= 0 and snr_db < 10:
            estimated_throughput = cc * 0.1
        else:
            estimated_throughput = 0
        return estimated_throughput
    
    def calculateBER(self, dist):
        if dist < self.maximum_radius:
            ber = self.predef_ber[0] - (((self.predef_ber[0] - self.predef_ber[1])/self.maximum_radius) * dist)
        return random.uniform(ber*0.9,ber*1.1)
        
    def calculateFEC(self, dist):
        if dist < self.maximum_radius:
            fec = self.predef_fec[0] - (((self.predef_fec[0] - self.predef_fec[1])/self.maximum_radius) * dist)
        return random.uniform(fec*0.9, fec*1.1)
    
    def calculateQoSParameters(self, device):
        QoS_Parameters = {}
        
        # Calculate Distance
        distance = self.calculateDeviceDistance(device)
        QoS_Parameters['Distance'] = round(distance, 3)
        
        # Calculate Free Space Path Loss
        fspl = self.calculateFSPL_db(distance)
        #QoS_Parameters['FSPL'] = round(fspl, 3)
        
        # Calculate Rayleigh Fading dB
        if self.fading == "Rayleigh": fading_value = self.calculateRayleighFading_db()
        
        # Calculate Rayleigh Fading dB
        if self.fading == "Rician": fading_value = -self.calculateRicianFading_db(5)
        
        # Add Fading Loss into FSPL
        if self.fading != None: fspl = fspl-fading_value
        
        # Add Shadowing into FSPL
        shadowing = self.calculateShadowing()
        fspl = fspl-shadowing
        
        # Calculate Received Signal Strength Indicator
        rssi = self.calculateRSSI_dbm(fspl)
        QoS_Parameters['RSSI'] = round(rssi, 3)
        
        # Calculate Signal Noise Ratio
        thermal_noise_dbm = self.calculateThermalNoise_dbm(5)
        snr_db = self.calculateSNR_db(rssi, thermal_noise_dbm)
        QoS_Parameters['SNR'] = round(snr_db, 3)
        
        # Calculate Channel Capacity
        channel_capacity = self.calculateChannelCapacity(snr_db)
        #QoS_Parameters['CC'] = round(channel_capacity/1000000, 3)
        
        # Calculate Estimated Throughput
        estimated_throughput = self.estimateThroughput(snr_db, channel_capacity)
        QoS_Parameters['Throughput'] = round(estimated_throughput/1000000, 3)
        
        # Verify Status
        network_status = self.calculateMinimumSNR_db(snr_db)
        
        # Update Max Radius Range
        if self.fading != None: self.update_transmission_range(fading_value+shadowing)
        
        if network_status == "Offline":
            QoS_Parameters = {}
            QoS_Parameters['Status'] = "Offline"
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}
        
        else:
            # Calculate BER - Pre-Defined
            ber = self.calculateBER(distance)
            QoS_Parameters['BER'] = ber
            
            # Calculate FEC - Pre-Defined
            fec = self.calculateFEC(distance)
            QoS_Parameters['FEC'] = fec
            
            # Add Protocol Parameters into QoS package
            QoS_Parameters['Protocol'] = self.protocol
            QoS_Parameters['PC'] = self.power_consumption
            QoS_Parameters['MC'] = self.monetary_cost
            
        
            QoS_Parameters = {**{'Status': 'Online'}, **QoS_Parameters}
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}

        return QoS_Parameters
    
    
    # ============================== Start Predef ==============================
    def transmission_range_predef(self):
        transmission_range_radius = []
        transmission_range_radius.append(self.maximum_radius)
        transmission_range_radius.append(2*self.maximum_radius/3)
        transmission_range_radius.append(self.maximum_radius/3)
        return transmission_range_radius
        
    def calculateMaximumRadius(self, dist):
        if dist > self.maximum_radius:
            status = "Offline"
        else:
            status = "Online"
        return status
        
    def estimateThroughput_predef(self, dist):
        if dist < self.maximum_radius:
            estimated_throughput = self.predef_throughput[0] - (((self.predef_throughput[0] - self.predef_throughput[1])/self.maximum_radius) * dist)
        return round(random.uniform(estimated_throughput*0.9,estimated_throughput*1.1), 2)
    
    def calculateSNR_predef(self, dist):
        if dist < self.maximum_radius:
            snr_db = self.predef_snr[0] - (((self.predef_snr[0] - self.predef_snr[1])/self.maximum_radius) * dist)
        return round(random.uniform(snr_db*0.9,snr_db*1.1), 2)
    
    def calculateRSSI_predef(self, dist):
        if dist < self.maximum_radius:
            rssi = self.predef_rssi[0] - (((self.predef_rssi[0] - self.predef_rssi[1])/self.maximum_radius) * dist)
        return round(random.uniform(rssi*0.9,rssi*1.1), 2)
        
    def calculateBER_predef(self, dist):
        if dist < self.maximum_radius:
            ber = self.predef_ber[0] - (((self.predef_ber[0] - self.predef_ber[1])/self.maximum_radius) * dist)
        return random.uniform(ber*0.9,ber*1.1)
        
    def calculateFEC_predef(self, dist):
        if dist < self.maximum_radius:
            fec = self.predef_fec[0] - (((self.predef_fec[0] - self.predef_fec[1])/self.maximum_radius) * dist)
        return random.uniform(fec*0.9, fec*1.1)
    
    def calculateQoSParametersPredef(self, device):
        QoS_Parameters = {}
        
        # Calculate Distance
        distance = self.calculateDeviceDistance(device)
        #print(f"Network: {self.system_name} || Distance: {distance} || x,y: {self.x_position},{self.y_position} || Device x,y: {device.x_position},{device.y_position}")
        QoS_Parameters['Distance'] = distance
        
        # Verify Status
        network_status = self.calculateMaximumRadius(distance)
        if network_status == "Offline":
            QoS_Parameters = {}
            QoS_Parameters['Status'] = "Offline"
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}
        else:
            # Calculate Received Signal Strength Indicator - Pre-Defined
            rssi = self.calculateRSSI_predef(distance)
            QoS_Parameters['RSSI'] = rssi
            
            # Calculate Signal Noise Ratio - Pre-Defined
            snr_db = self.calculateSNR_predef(distance)
            QoS_Parameters['SNR'] = snr_db
            
            # Calculate BER - Pre-Defined
            ber = self.calculateBER_predef(distance)
            QoS_Parameters['BER'] = ber
            
            # Calculate FEC - Pre-Defined
            fec = self.calculateFEC_predef(distance)
            QoS_Parameters['FEC'] = fec
            
            # Calculate Estimated Throughput - Pre-Defined
            estimated_throughput = self.estimateThroughput_predef(distance)
            QoS_Parameters['Throughput'] = round(estimated_throughput/1000000, 3)
            
            # Add Protocol Parameters into QoS package
            QoS_Parameters['Protocol'] = self.protocol
            QoS_Parameters['PC'] = self.power_consumption
            QoS_Parameters['MC'] = self.monetary_cost
            
            QoS_Parameters = {**{'Status': 'Online'}, **QoS_Parameters}
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}

        return QoS_Parameters
    # ============================== End Predef ==============================
        
        
    def __repr__(self):
        return f"WirelessNetworkSystem: ({self.system_name}, X: {self.x_position}, Y: {self.y_position})"