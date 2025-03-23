class Device:
    
    def __init__(self, device_id, x_position, y_position):
        self.device_id = device_id
        self.x_position = x_position
        self.y_position = y_position
        self.networks = set()
    
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
    
    def __repr__(self):
        return f"Device: ({self.device_id}, X: {self.x_position}, Y: {self.y_position}, WirelessNetworkConnected: {self.networks})"