class Device:
    
    def __init__(self, device_id, x_position, y_position):
        self.device_id = device_id
        self.x_position = x_position
        self.y_position = y_position
        self.networks = set()
        self.networks_list = []
        self.available_networks = []
        #self.decision_maker_methods = set()
        
        '''
        dict_now = {}
        dict_now['MPMO-Fuzzy'] = 0
        dict_now['MPMO-TOPSIS'] = 0
        dict_now['MPMO-WPM'] = 0
        dict_now['MPMO-SAW'] = 0
        
        self.dict_now = dict_now
        '''
    
    def updatePosition(self, x, y):
        self.x_position = x
        self.y_position = y
    
    def connect_to_network(self, network):
        # Attach the device to a network 
        network.attach_device(self)
     
    def disconnect_from_network(self, network):
        # Detach the device from a network
        network.detach_device(self)
    
    def get_QoS_Parameters(self, network):
        if network in self.networks:
            QoS_Parameters = network.calculateQoSParameters(self)
            return QoS_Parameters
            
    def get_all_QoS_Parameters(self):
        self.networks_list = []
        for network in self.networks:
            QoS_Parameters = network.calculateQoSParameters(self)
            self.networks_list.append(QoS_Parameters)
        return self.networks_list
    
    # ============================== Start Predef ==============================
    def get_QoS_Parameters_predef(self, network):
        if network in self.networks:
            QoS_Parameters = network.calculateQoSParametersPredef(self)
            return QoS_Parameters
            
    def get_all_QoS_Parameters_predef(self):
        self.networks_list = []
        for network in self.networks:
            QoS_Parameters = network.calculateQoSParametersPredef(self)
            self.networks_list.append(QoS_Parameters)
        return self.networks_list
    # ============================== End Predef ==============================
    
    def get_available_networks(self):
        self.available_networks = []
        for network in self.networks_list: 
            if network['Status'] == 'Online':
                self.available_networks.append(network)
        return self.available_networks
        
    def makeDecision(self, decision_maker_method, inputs):
        decision_maker_method.send_inputs(inputs)
        '''if decision_maker_method.method_name == "MPMO-Fuzzy" or decision_maker_method.method_name == "MPMO-TOPSIS" or decision_maker_method.method_name == "MPMO-WPM" or decision_maker_method.method_name == "MPMO-SAW":
            #print(f"DM Method: {decision_maker_method.method_name}")
            start_time = time.perf_counter()
            #print(f"Start Time: {start_time}")'''
        output = decision_maker_method.makeDecision()
        '''if decision_maker_method.method_name == "MPMO-Fuzzy" or decision_maker_method.method_name == "MPMO-TOPSIS" or decision_maker_method.method_name == "MPMO-WPM" or decision_maker_method.method_name == "MPMO-SAW":
            end_time = time.perf_counter()
            #print(f"End Time: {end_time}")
            diff = end_time-start_time
            #print(f"Difference: {diff}")
            self.dict_now[decision_maker_method.method_name] = self.dict_now[decision_maker_method.method_name] + diff
            print("==================================")
            print(self.dict_now)'''
        return output
            
    def __repr__(self):
        return f"Device: ({self.device_id}, X: {self.x_position}, Y: {self.y_position}, WirelessNetworkConnected: {self.networks})"