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
        self.old_dist = {}
        self.outage = 0
        self.hp = None
        self.hf_too_late = 0      # radio link failures (current network lost)
        self.hf_execution = 0     # failed handover attempts
    
    def resetParameters(self):
        self.inputs = None
        self.inputs_bkp = None
        self.output = None
        self.old_decision = None
        self.old_dist = {}
        self.outage = 0
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
                            input['HC'] = 0.5
                        elif input['Protocol'] == "WiFi-5GHz":
                            input['HC'] = 0.4
                        elif input['Protocol'] == "NB-IoT":
                            input['HC'] = 0.75
                        elif input['Protocol'] == "LoRa-868":
                            input['HC'] = 0.95
                        elif input['Protocol'] == "LTE-4G":
                            input['HC'] = 0.65
                        elif input['Protocol'] == "5G-N78":
                            input['HC'] = 0.25
                        elif input['Protocol'] == "5G-N28":
                            input['HC'] = 0.3
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
    
    '''def return_output(self):

        if self.hp:
            final_output = next((inp for inp in self.inputs_bkp if inp['Network'] == self.output['Network']), None)
            
            #print("==================== INP BKP 1 ====================")
            #print(self.inputs_bkp)
            #print("=================================================")
            
            HF_PROBABILITY_TABLE_Too_Early = {
                "WiFi-2.4GHz": 0.05, "WiFi-5GHz": 0.03, "NB-IoT": 0.12,
                "LoRa-868": 0.18, "LTE-4G": 0.04, "5G-N78": 0.01, "5G-N28": 0.02,
            }

            HF_PROBABILITY_TABLE_Too_Late = {
                "WiFi-2.4GHz": 0.05, "WiFi-5GHz": 0.03, "NB-IoT": 0.12,
                "LoRa-868": 0.18, "LTE-4G": 0.04, "5G-N78": 0.01, "5G-N28": 0.02,
            }

            protocol_minimum_snr = {
                "WiFi-2.4GHz": 10, "WiFi-5GHz": 7, "NB-IoT": 2,
                "LoRa-868": 0, "LTE-4G": 5, "5G-N78": 4, "5G-N28": 3,
            }
            
            disconnected = {
                'Network': 'Offline', 
                'Status': 'Offline', 
                'Distance': np.float64(0), 
                'RSSI': np.float64(0), 
                'SNR': 0, 
                'Throughput': 0, 
                'BER': np.float64(0.01), 
                'FEC': np.float64(0.5), 
                'Protocol': 'Offline', 
                'PC': 1.5, 
                'MC': 5, 
                'Delay': np.float64(1500), 
                'Jitter': np.float64(600), 
                'HC': 1
            }
            
            if self.old_decision != None and final_output['Network'] != self.old_decision:
                if final_output['Protocol'] == "WiFi-2.4GHz":
                    final_output['HC'] = 0.5
                elif final_output['Protocol'] == "WiFi-5GHz":
                    final_output['HC'] = 0.4
                elif final_output['Protocol'] == "NB-IoT":
                    final_output['HC'] = 0.75
                elif final_output['Protocol'] == "LoRa-868":
                    final_output['HC'] = 0.95
                elif final_output['Protocol'] == "LTE-4G":
                    final_output['HC'] = 0.65
                elif final_output['Protocol'] == "5G-N78":
                    final_output['HC'] = 0.25
                elif final_output['Protocol'] == "5G-N28":
                    final_output['HC'] = 0.3
                else:
                    final_output['HC'] = 1

                dist_diff = 0
                prev_dist = self.old_dist.get(final_output['Network'])
                if prev_dist is not None:
                    dist_diff = final_output['Distance'] - prev_dist


                EDGE_BAND = 10.0      # dB above minimum SNR where the risk starts to grow
                MAX_MULT = 10.0       # multiplier at the edge when moving directly away
                REF_SPEED = 1.5       # m per step (15 m/s car); receding at this speed = full effect

                proto = final_output['Protocol']
                margin = final_output['SNR'] - protocol_minimum_snr[proto]

                # 0 = far from the edge, 1 = at the edge
                closeness = min(max(1 - margin / EDGE_BAND, 0.0), 1.0)

                # 0 = approaching or static, 1 = moving away at full speed
                receding = min(max(dist_diff / REF_SPEED, 0.0), 1.0)

                mult = 1 + (MAX_MULT - 1) * closeness * receding

                p_early = min(HF_PROBABILITY_TABLE_Too_Early[proto] * mult, 0.9)
                p_late = min(HF_PROBABILITY_TABLE_Too_Late[proto] * mult, 0.9)

                r = random.random()
                if r < p_early:
                    final_output = copy.deepcopy(next((inp for inp in self.inputs_bkp if inp['Network'] == self.old_decision), disconnected))
                    final_output['HC'] = 1
                elif r < p_early + p_late:
                    final_output = copy.deepcopy(disconnected)
                    final_output['HC'] = 1
                    
            #print("==================== INP BKP 2 ====================")
            #print(self.old_decision)
            #print(final_output)
            #print("=================================================")
        else:
            final_output = self.output
        self.inputs = self.inputs_bkp


        if self.hp:
            self.old_dist = {}
            for input in self.inputs_bkp:
                self.old_dist[input['Network']] = input['Distance']

        return final_output'''

    def return_output(self):

        if self.hp:
            final_output = next((inp for inp in self.inputs_bkp if inp['Network'] == self.output['Network']), None)
            
            HF_PROBABILITY_TABLE_Too_Early = {
                "WiFi-2.4GHz": 0.010, "WiFi-5GHz": 0.006, "NB-IoT": 0.024,
                "LoRa-868": 0.036, "LTE-4G": 0.008, "5G-N78": 0.002, "5G-N28": 0.004,
            }

            HF_PROBABILITY_TABLE_Too_Late = {
                "WiFi-2.4GHz": 0.010, "WiFi-5GHz": 0.006, "NB-IoT": 0.024,
                "LoRa-868": 0.036, "LTE-4G": 0.008, "5G-N78": 0.002, "5G-N28": 0.004,
            }

            protocol_minimum_snr = {
                "WiFi-2.4GHz": 10, "WiFi-5GHz": 7, "NB-IoT": 2,
                "LoRa-868": 0, "LTE-4G": 5, "5G-N78": 4, "5G-N28": 3,
            }
            
            disconnected = {
                'Network': 'Offline', 
                'Status': 'Offline', 
                'Distance': np.float64(0), 
                'RSSI': np.float64(-120),
                'SNR': 0, 
                'Throughput': 0, 
                'BER': np.float64(0.01), 
                'FEC': np.float64(0.1666), 
                'Protocol': 'Offline', 
                'PC': 1.5, 
                'MC': 5, 
                'Delay': np.float64(1500), 
                'Jitter': np.float64(600), 
                'HC': 1
            }

            OUTAGE_STEPS = 10       # 1 second of disconnection after a failed handover (10 x 100 ms)

            # Too-late handover: the current network disappeared before the device left it
            current_nets = {inp['Network'] for inp in self.inputs_bkp}
            if self.outage == 0 and self.old_decision not in (None, 'Offline') and self.old_decision not in current_nets:
                # Radio link failure: stayed too long on a network that went out of range
                self.outage = OUTAGE_STEPS
                self.hf_too_late += 1

            if self.outage > 0:
                # Still inside the 1 s outage from a failed handover
                self.outage -= 1
                final_output = copy.deepcopy(disconnected)

            elif self.old_decision != None and final_output['Network'] != self.old_decision:
                if final_output['Protocol'] == "WiFi-2.4GHz":
                    final_output['HC'] = 0.5
                elif final_output['Protocol'] == "WiFi-5GHz":
                    final_output['HC'] = 0.4
                elif final_output['Protocol'] == "NB-IoT":
                    final_output['HC'] = 0.75
                elif final_output['Protocol'] == "LoRa-868":
                    final_output['HC'] = 0.95
                elif final_output['Protocol'] == "LTE-4G":
                    final_output['HC'] = 0.65
                elif final_output['Protocol'] == "5G-N78":
                    final_output['HC'] = 0.25
                elif final_output['Protocol'] == "5G-N28":
                    final_output['HC'] = 0.3
                else:
                    final_output['HC'] = 1

                dist_diff = 0
                prev_dist = self.old_dist.get(final_output['Network'])
                if prev_dist is not None:
                    dist_diff = final_output['Distance'] - prev_dist

                # ================= Velocity-dependent handover failure =================
                APPROACH_SCALE = 0.1    # approaching the AP at full speed: failure 10x lower than base
                EDGE_BAND = 12.0        # dB above minimum SNR where the risk starts to grow
                P_MAX = 0.7             # total failure chance at the edge, moving straight away
                REF_SPEED = 1.5         # m per step (15 m/s car)

                proto = final_output['Protocol']
                table_early = HF_PROBABILITY_TABLE_Too_Early[proto]
                table_late = HF_PROBABILITY_TABLE_Too_Late[proto]
                base = table_early + table_late

                margin = final_output['SNR'] - protocol_minimum_snr[proto]
                closeness = min(max(1 - margin / EDGE_BAND, 0.0), 1.0)   # 0 = safe, 1 = at the edge
                speed = min(abs(dist_diff) / REF_SPEED, 1.0)             # 0 = static, 1 = full speed

                if dist_diff > 0:
                    # Moving away: risk grows toward P_MAX near the edge
                    p_fail = base + (P_MAX - base) * closeness * speed
                elif dist_diff < 0:
                    # Approaching: risk drops below the base
                    p_fail = base * (1 - (1 - APPROACH_SCALE) * speed)
                else:
                    # No history for this network: base risk
                    p_fail = base

                if random.random() < p_fail:
                    # Handover failed: disconnected now and for the rest of the 1 s outage
                    final_output = copy.deepcopy(disconnected)
                    self.outage = OUTAGE_STEPS - 1
                    self.hf_execution += 1
                # ========================================================================

        else:
            final_output = self.output
        self.inputs = self.inputs_bkp


        if self.hp:
            self.old_dist = {}
            for input in self.inputs_bkp:
                self.old_dist[input['Network']] = input['Distance']

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