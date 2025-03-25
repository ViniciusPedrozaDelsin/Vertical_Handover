import math
import random

class WirelessNetworkSystem:
    def __init__(self, system_name, x_position, y_position, transmission_power_dbm, frequency, bandwidth, minimum_snr, protocol, maximum_radius=None, predef_throughput=None, predef_snr=None, predef_rssi=None, corrections_real_world_applications=False):
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
        self.corrections_real_world_applications = corrections_real_world_applications
        
        # Pre-defined Configs
        self.maximum_radius = maximum_radius
        self.predef_throughput = predef_throughput
        self.predef_snr = predef_snr
        self.predef_rssi = predef_rssi
        
    
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
    
    def calculateDeviceDistance(self, device):
        distance = (((device.x_position - self.x_position)**2) + ((device.y_position - self.y_position))**2)**(1/2)
        return round(distance, 3)
        
    def calculateFSPL_db(self, d):
        # Distance correction if d = 0
        if d == 0: d = d + 0.00001
        # Speed of Light
        c = 300000000
        if self.corrections_real_world_applications == True:
            # Add Miscellaneous Loss = 20db
            free_space_path_loss_db = (20*math.log10(d)) + (20*math.log10(self.frequency)) + (20*math.log10((4*math.pi)/c)) + 20
            # FSPL correction if it is less then twenty
            if free_space_path_loss_db < 20: free_space_path_loss_db = 20
        else:
            # FSPL "Vanilla"
            free_space_path_loss_db = (20*math.log10(d)) + (20*math.log10(self.frequency)) + (20*math.log10((4*math.pi)/c))
            # FSPL correction if it is less then zero
            if free_space_path_loss_db < 0: free_space_path_loss_db = 0
        return round(free_space_path_loss_db, 3)
        
    def calculateCOST231HataModel(self, d):
        pass
        
    def calculateRSSI_dbm(self, fspl):
        rssi_dbm = self.transmission_power_dbm - fspl
        return round(rssi_dbm, 3)
        
    def calculateThermalNoise_dbm(self):
        # Boltzmann's Constant (J/K)
        k = 1.38 * (10**-23)
        # Temperature 290 Kelvin
        T = 290
        # Thermal Noise Formula
        thermal_noise_power_dbm = 10 * math.log10(k*T*self.bandwidth) + 30
        return thermal_noise_power_dbm
       
    def calculateSNR_db(self, rssi_dbm, thermal_noise_power_dbm):
        snr = (10**(rssi_dbm/10)) / (10**(thermal_noise_power_dbm/10))
        if self.corrections_real_world_applications == True:
            # Correction of 0.7 in the SNR for real world applications
            snr_db = 10 * math.log10(snr*0.5)
        else:
            snr_db = 10 * math.log10(snr)
        return round(snr_db, 3)
    
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
    
    
    def calculateQoSParameters(self, device):
        QoS_Parameters = {}
        
        # Calculate Distance
        distance = self.calculateDeviceDistance(device)
        QoS_Parameters['Distance'] = distance
        
        # Calculate Free Space Path Loss
        fspl = self.calculateFSPL_db(distance)
        QoS_Parameters['FSPL'] = fspl
        
        # Calculate Received Signal Strength Indicator
        rssi = self.calculateRSSI_dbm(fspl)
        QoS_Parameters['RSSI'] = rssi
        
        # Calculate Signal Noise Ratio
        thermal_noise_dbm = self.calculateThermalNoise_dbm()
        snr_db = self.calculateSNR_db(rssi, thermal_noise_dbm)
        QoS_Parameters['SNR'] = snr_db
        
        # Calculate Channel Capacity
        channel_capacity = self.calculateChannelCapacity(snr_db)
        QoS_Parameters['CC'] = round(channel_capacity/1000000, 3)
        
        # Calculate Estimated Throughput
        estimated_throughput = self.estimateThroughput(snr_db, channel_capacity)
        QoS_Parameters['Throughput'] = round(estimated_throughput/1000000, 3)
        
        # Add Protocol into QoS package
        QoS_Parameters['Protocol'] = self.protocol
        
        # Verify Status
        network_status = self.calculateMinimumSNR_db(snr_db)
        if network_status == "Offline":
            QoS_Parameters = {}
            QoS_Parameters['Status'] = "Offline"
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}
        else:
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
        if dist < self.maximum_radius and dist >= ((2*self.maximum_radius)/3):
            estimated_throughput = self.predef_throughput[2]
        elif dist < ((2*self.maximum_radius)/3) and dist >= (self.maximum_radius/3):
            estimated_throughput = self.predef_throughput[1]
        else:
            estimated_throughput = self.predef_throughput[0]
        return round(random.uniform(estimated_throughput*0.9,estimated_throughput*1.1), 2)
    
    def calculateSNR_predef(self, dist):
        if dist < self.maximum_radius and dist >= ((2*self.maximum_radius)/3):
            snr_db = self.predef_snr[2]
        elif dist < ((2*self.maximum_radius)/3) and dist >= (self.maximum_radius/3):
            snr_db = self.predef_snr[1]
        else:
            snr_db = self.predef_snr[0]
        return round(random.uniform(snr_db*0.9,snr_db*1.1), 2)
    
    def calculateRSSI_predef(self, dist):
        if dist < self.maximum_radius and dist >= ((2*self.maximum_radius)/3):
            rssi = self.predef_rssi[2]
        elif dist < ((2*self.maximum_radius)/3) and dist >= (self.maximum_radius/3):
            rssi = self.predef_rssi[1]
        else:
            rssi = self.predef_rssi[0]
        return round(random.uniform(rssi*0.9,rssi*1.1), 2)
    
    def calculateQoSParametersPredef(self, device):
        QoS_Parameters = {}
        
        # Calculate Distance
        distance = self.calculateDeviceDistance(device)
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
            
            # Calculate Estimated Throughput - Pre-Defined
            estimated_throughput = self.estimateThroughput_predef(distance)
            QoS_Parameters['Throughput'] = round(estimated_throughput/1000000, 3)
            
            QoS_Parameters['Protocol'] = self.protocol
            
            QoS_Parameters = {**{'Status': 'Online'}, **QoS_Parameters}
            QoS_Parameters = {**{'Network': self.system_name}, **QoS_Parameters}
        
        return QoS_Parameters
    # ============================== End Predef ==============================
        
        
    def __repr__(self):
        return f"WirelessNetworkSystem: ({self.system_name}, X: {self.x_position}, Y: {self.y_position})"