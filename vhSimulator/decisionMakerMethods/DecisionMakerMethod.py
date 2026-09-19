import pandas as pd
import numpy as np
import hashlib
import random
import copy

class DecisionMakerMethod:

    def __init__(self, method_name, file_to_save=None):
        self.method_name = method_name
        self.file_to_save = file_to_save
        self.inputs = None
        self.inputs_bkp = None
        self.output = None
        self.old_decision = None
        self.hp = None
    
    def resetParameters(self):
        self.inputs = None
        self.inputs_bkp = None
        self.output = None
        self.old_decision = None
        self.hp = None
        
    def send_inputs(self, inputs, hp=False):
        self.hp = hp
        if self.hp:
            self.inputs_bkp = copy.deepcopy(inputs) 
            #print("==================== INP BKP 1 ====================")
            #print(self.inputs_bkp)
            #print("=================================================")
            if self.old_decision != None:
                for input in inputs:
                    if input['Network'] != self.old_decision:
                        #input['Delay'] = 510
                        #input['Jitter'] = 150
                        input['Delay'] = 415.7142857
                        input['Jitter'] = 122.142857
                        input['HC'] = 1
                        if input['Protocol'] == "WiFi-2.4GHz":
                            input['HC'] = 0.22
                        elif input['Protocol'] == "WiFi-5GHz":
                            input['HC'] = 0.15
                        elif input['Protocol'] == "NB-IoT":
                            input['HC'] = 0.75
                        elif input['Protocol'] == "LoRa-868":
                            input['HC'] = 0.9
                        elif input['Protocol'] == "LTE-4G":
                            input['HC'] = 0.25
                        elif input['Protocol'] == "5G-N78":
                            input['HC'] = 0.05
                        elif input['Protocol'] == "5G-N28":
                            input['HC'] = 0.1
                        else:
                            input['HC'] = 1
                #print("==================================== Inputs ====================================")        
                self.inputs = inputs
                #print(self.inputs)
                #print("==================================== '' ====================================")
            else:
                for input in inputs:
                    #input['Delay'] = 510
                    #input['Jitter'] = 150
                    input['Delay'] = 415.7142857
                    input['Jitter'] = 122.142857
                self.inputs = inputs
        else:
            self.inputs = inputs
    
    def return_output(self):
        if self.hp:
            final_output = next((inp for inp in self.inputs_bkp if inp['Network'] == self.output['Network']), None)
            
            #print("==================== INP BKP 1 ====================")
            #print(self.inputs_bkp)
            #print("=================================================")
            
            HF_PROBABILITY_TABLE = {
                "WiFi-2.4GHz": 0.05, "WiFi-5GHz": 0.03, "NB-IoT": 0.12,
                "LoRa-868": 0.18, "LTE-4G": 0.04, "5G-N78": 0.01, "5G-N28": 0.02,
            }
            
            disconnected = {
                'Network': 'Disconnected', 
                'Status': 'Offline', 
                'Distance': np.float64(0), 
                'RSSI': np.float64(0), 
                'SNR': 0, 
                'Throughput': 0, 
                'BER': np.float64(0.01), 
                'FEC': np.float64(0.5), 
                'Protocol': 'Disconnected', 
                'PC': 1.5, 
                'MC': 5, 
                'Delay': np.float64(1500), 
                'Jitter': np.float64(600), 
                'HC': 1
            }
            
            if self.old_decision != None and final_output['Network'] != self.old_decision:
                if final_output['Protocol'] == "WiFi-2.4GHz":
                    final_output['HC'] = 0.22
                elif final_output['Protocol'] == "WiFi-5GHz":
                    final_output['HC'] = 0.15
                elif final_output['Protocol'] == "NB-IoT":
                    final_output['HC'] = 0.75
                elif final_output['Protocol'] == "LoRa-868":
                    final_output['HC'] = 0.9
                elif final_output['Protocol'] == "LTE-4G":
                    final_output['HC'] = 0.25
                elif final_output['Protocol'] == "5G-N78":
                    final_output['HC'] = 0.05
                elif final_output['Protocol'] == "5G-N28":
                    final_output['HC'] = 0.1
                else:
                    final_output['HC'] = 1
                
                if random.random() < HF_PROBABILITY_TABLE[final_output['Protocol']]:
                    final_output = next((inp for inp in self.inputs_bkp if inp['Network'] == self.old_decision), disconnected)
                    final_output['HC'] = 1
                    #print(self.old_decision)
                    #print(final_output)
                    
            #print("==================== INP BKP 2 ====================")
            #print(self.old_decision)
            #print(final_output)
            #print("=================================================")
        else:
            final_output = self.output
        self.inputs = self.inputs_bkp
        return final_output
    
    def create_unique_id(*args):
        # Combine all inputs into a single string
        combined = '_'.join(map(str, args))
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def saveData(self, outputs):
        #print(self.inputs)
        #print(outputs)
        
        # Fields to normalize
        fields = ['RSSI', 'SNR', 'BER', 'FEC', 'Throughput', 'PC', 'MC', 'Delay', 'Jitter', 'HC']

        # Compute min and max for each field
        mins = {field: min(d[field] for d in self.inputs) for field in fields}
        maxs = {field: max(d[field] for d in self.inputs) for field in fields}

        # Normalize
        normalized_data = []
        for item in self.inputs:
            normalized_item = item.copy()
            for field in fields:
                min_val = mins[field]
                max_val = maxs[field]
                if max_val == min_val:
                    normalized_item[field] = 0.0
                else:
                    normalized_item[field] = (item[field] - min_val) / (max_val - min_val)
            normalized_data.append(normalized_item)
            
        uuid = self.create_unique_id(self.inputs, outputs)
        data = []
        i = 0
        for input_network in normalized_data:
            data.append(
                [
                    uuid,
                    #input_network['Network'], 
                    #input_network['Status'], 
                    #input_network['Distance'], 
                    input_network['RSSI'], 
                    input_network['SNR'], 
                    input_network['Throughput'], 
                    input_network['BER'], 
                    input_network['FEC'], 
                    #input_network['Protocol'], 
                    input_network['PC'], 
                    input_network['MC'], 
                    input_network['Delay'],
                    input_network['Jitter'],
                    input_network['HC'],
                    outputs[i]
                ]
            )
            i += 1
        df = pd.DataFrame(data)
        df.to_csv(self.file_to_save, mode='a', header=False, index=False)
        
    # ================================ LockIn: Only available for MPMO
    def check_lockin_reference(self):
        network_still_available = False
        actual_network = None
        if self.lockin_reference != None:
            for input in self.inputs:
                if self.lockin_reference['Network'] == input['Network']:
                    # Get new parameters of the actual network
                    actual_network = input
                    network_still_available = True
        
        network_scanning = [True, None]
        if network_still_available:
            network_ok = True
            i = 0
            for attribute in self.attributes:
                if self.directions[i] == 1 and self.lockin_reference[attribute] > 0:
                    if actual_network[attribute] <= self.lockin_reference[attribute] * (1 - self.lockin_percentage):
                        network_ok = False
                elif self.directions[i] == 1 and self.lockin_reference[attribute] < 0:
                    if actual_network[attribute] <= self.lockin_reference[attribute] * (1 + self.lockin_percentage):
                        network_ok = False
                elif self.directions[i] == 0 and self.lockin_reference[attribute] > 0:
                    if actual_network[attribute] >= self.lockin_reference[attribute] * (1 + self.lockin_percentage):
                        network_ok = False
                else:
                    if actual_network[attribute] >= self.lockin_reference[attribute] * (1 - self.lockin_percentage):
                        network_ok = False
                i = i + 1
            if network_ok:
                network_scanning = [False, actual_network]
            
        return network_scanning
    
        
    # ================================ Time to Trigger (TTT): Only available for MPMO    
    def check_time_to_trigger(self, ex_output):
        if self.ttt_reference is None or self.ttt_active_network is None:
            output = ex_output
        else:
            if self.ttt_active_network['Network'] == ex_output['Network']:
                output = ex_output
                self.actual_ttt = 0
            else:
                if self.ttt_reference['Network'] == ex_output['Network']:
                    if self.actual_ttt >= self.time_to_trigger-1:
                        output = ex_output
                        self.actual_ttt = 0
                    else:
                        # Set expected output as output except if there is the active network available
                        output = ex_output
                        for ipt in self.inputs:
                            if self.ttt_active_network['Network'] == ipt['Network']:
                                output = self.ttt_active_network
                        self.actual_ttt += 1
                else:
                    # Set expected output as output except if there is the active network available
                    output = ex_output
                    if self.time_to_trigger != 0:
                        for ipt in self.inputs:
                            if self.ttt_active_network['Network'] == ipt['Network']:
                                output = self.ttt_active_network
                    self.actual_ttt = 0
            
        self.ttt_reference = ex_output
        self.ttt_active_network = output
        return output