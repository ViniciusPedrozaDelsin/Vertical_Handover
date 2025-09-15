from .DecisionMakerMethod import DecisionMakerMethod as DMM

class MPMO_Fuzzy(DMM):
    def __init__(self, method_name, attributes=None, directions=None, hysterese_percentage=None, lockin_percentage=None, time_to_trigger=None):
        super().__init__(method_name)
        self.attributes = attributes
        self.directions = directions
        self.membership = {}
        self.membership_degree = {}
        self.rules = []
        
        # Hysterese Value
        self.hysterese_percentage = hysterese_percentage
        
        # LockIn values
        self.lockin_reference = None
        self.lockin_percentage = lockin_percentage
        
        # Time to Trigger values
        self.actual_ttt = 0
        self.ttt_reference = None
        self.ttt_active_network = None
        self.time_to_trigger = time_to_trigger
        
    
    def makeDecision(self):
        if self.lockin_percentage != None:
            self.output = self.makeDecisionLockin()
        elif self.time_to_trigger != None:
            self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()
            
        self.old_decision = self.output['Network']
        self.output = self.return_output()
        return self.output
    
    
    def makeDecisionLockin(self):
        check_lockin_reference = self.check_lockin_reference()
        if check_lockin_reference[0]:
            self.output = self.decisionProcedure()
            self.lockin_reference = self.output
        else:
            self.output = check_lockin_reference[1]
        return self.output
        
    def makeDecisionTimeToTrigger(self):
        expected_output = self.decisionProcedure()
        self.output = self.check_time_to_trigger(expected_output)
        return self.output
    
    def decisionProcedure(self):
        fuzzification = self.Fuzzification(self.membership_degree)
        applied_rules = self.ApplyRules(fuzzification)
        final_results = self.Defuzzification(applied_rules)
        output = self.choose_handoff_fuzzified(final_results)
        #output = {'Network': 'LoRa-2', 'Status': 'Online', 'Distance': 7710.096427074727, 'RSSI': -101.02, 'SNR': 2.37, 'BER': 0.008294665075988363, 'FEC': 0.5765109730288961, 'Throughput': 0.013, 'Protocol': 'LoRa-868', 'PC': 0.05, 'MC': 1}
        return output
    
    def defineMembershipDegree(self):
        pass
    
    def defineRules(self):
        pass
    
    def definePresetConfigs(self):
        # Set Membership Degree
        self.membership['RSSI'] = {'Low': [-150, -80], 'Medium': [-100, -60], 'High': [-75, -35]}
        self.membership['SNR'] = {'Low': [0, 15], 'High': [11, 30]}
        self.membership['Throughput'] = {'Low': [0, 15], 'Medium': [10, 100], 'High': [50, 1500]}
        self.membership['PC'] = {'Low': [0.05, 0.6], 'High': [0.4, 1.05]}
        self.membership['MC'] = {'Low': [1, 2.5], 'High': [2, 4]}
        self.membership['BER'] = {'Low': [0.00000001, 0.00001], 'High': [0.000001, 0.01]}
        self.membership['FEC'] = {'Low': [0.5, 0.75], 'High': [0.7, 1]}
        
        # Set Fuzzy Rules
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'OR', 'Rules': {'Throughput': 'High', 'SNR': 'High', 'RSSI': 'High', 'PC': 'Low', 'MC': 'Low', 'BER': 'Low', 'FEC': 'Low'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'Throughput': 'Medium', 'SNR': 'High', 'PC': 'Low', 'MC': 'Low'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'Throughput': 'High', 'RSSI': 'Medium', 'FEC': 'High'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'Throughput': 'High', 'RSSI': 'High'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'OR', 'Rules': {'Throughput': 'High', 'SNR': 'High'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'RSSI': 'High', 'SNR': 'High', 'FEC': 'High'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'Throughput': 'High', 'PC': 'Low', 'MC': 'Low'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'OR', 'Rules': {'SNR': 'High', 'BER': 'Low'}})
        self.rules.append({'Handoff': 'YES', 'Conjunction': 'AND', 'Rules': {'Throughput': 'Medium', 'BER': 'Low'}})
        self.rules.append({'Handoff': 'NO', 'Conjunction': 'AND', 'Rules': {'Throughput': 'Low', 'BER': 'High'}})
        self.rules.append({'Handoff': 'NO', 'Conjunction': 'OR', 'Rules': {'Throughput': 'Low', 'PC': 'High'}})
        self.rules.append({'Handoff': 'NO', 'Conjunction': 'AND', 'Rules': {'RSSI': 'Low', 'PC': 'High', 'FEC': 'Low'}})
        self.rules.append({'Handoff': 'NO', 'Conjunction': 'OR', 'Rules': {'Throughput': 'Low', 'SNR': 'Low'}})
        self.rules.append({'Handoff': 'NO', 'Conjunction': 'AND', 'Rules': {'RSSI': 'Low', 'SNR': 'Low'}})
        
        self.calculateTrianglesPeak()
    
    def calculateTrianglesPeak(self):
        self.membership_degree = self.membership
        for key, value in self.membership.items():
            for name, v in value.items():
                v_sum = 0
                for element in v:
                    v_sum = v_sum + element
                self.membership_degree[key][name].insert(1, v_sum/2)
        return self.membership_degree
    
    def calculateTrianglesSlope(self, parameter, value):
        output = {}
        for key, value_mdegree in self.membership_degree[parameter].items():
            if key == "High":
                if value >= value_mdegree[2]:
                    output[key] = 1
                elif value <= value_mdegree[0]:
                    output[key] = 0
                else:
                    output[key] = (value - value_mdegree[0]) / (value_mdegree[2] - value_mdegree[0])
            elif key == "Low":
                if value >= value_mdegree[2]:
                    output[key] = 0
                elif value <= value_mdegree[0]:
                    output[key] = 1
                else:
                    output[key] = (value_mdegree[2] - value) / (value_mdegree[2] - value_mdegree[0])
            else:
                if value >= value_mdegree[2] or value <= value_mdegree[0]:
                    output[key] = 0
                else:
                    # Left Slope
                    if value < value_mdegree[1]:
                        output[key] = (value - value_mdegree[0]) / (value_mdegree[1] - value_mdegree[0])
                        
                    # Right Slope
                    elif value > value_mdegree[1]:
                        output[key] = (value_mdegree[2] - value) / (value_mdegree[2] - value_mdegree[1])
                        
                    # Triangle Peak
                    else:
                        output[key] = 1
            
        return output
    
    
    def Fuzzification(self, membership_degree):
        fuzzified_inputs = []
        for ipt in self.inputs:
            fuzzified_input = {}
            fuzzified_input['Network'] = ipt['Network']
            for att, value in ipt.items():
                if att in self.membership_degree:
                    fuzzified_input[att] = self.calculateTrianglesSlope(att, value)
            fuzzified_inputs.append(fuzzified_input)
        return fuzzified_inputs
    
    
    def ApplyRules(self, fuzzified_inputs):
        for fuzzified_input in fuzzified_inputs:
            rules_dict = {}
            rules_dict['YES'] = []
            rules_dict['NO'] = []
            for rule in self.rules:
                activation_values  = []
                for k, v in rule['Rules'].items():
                    activation_values.append(fuzzified_input[k][v])
                if rule['Conjunction'] == 'AND':
                    activation_strength  = min(activation_values)
                else:
                    activation_strength  = max(activation_values)
                
                if rule['Handoff'] == "YES":
                    rules_dict['YES'].append(activation_strength)
                else:
                    rules_dict['NO'].append(activation_strength)
            fuzzified_input['Strength'] = rules_dict
        return fuzzified_inputs
    
    def Defuzzification(self, fuzzified_values):
        for fv in fuzzified_values:
            sum_yes = sum(fv['Strength']['YES'])
            sum_no = sum(fv['Strength']['NO'])
            crisp_output = sum_yes / (sum_yes + sum_no)
            fv['CrispOutput'] = crisp_output
        return fuzzified_values
    
    def choose_handoff_fuzzified(self, crisp_results):
        crisp_resp = 0
        choosen_network = None
        output = None
        for cr in crisp_results:
            #print(f"{cr['CrispOutput']} || {cr['Network']}")
            if cr['CrispOutput'] >= crisp_resp:
                crisp_resp = cr['CrispOutput']
                choosen_network = cr['Network']
        
        for inpt in self.inputs:
            if inpt['Network'] == choosen_network:
                output = inpt
        
        # Hysteresis
        if self.hysterese_percentage != None:
            networks = [net['Network'] for net in self.inputs]
            if self.old_decision in networks:
                if self.old_decision != output['Network']:
                    for cr in crisp_results:
                        if cr['Network'] == self.old_decision:
                            old_crisp = cr['CrispOutput']
                    if old_crisp * (1 + self.hysterese_percentage) >= crisp_resp:
                        output = self.inputs[networks.index(self.old_decision)]
                
        return output
    