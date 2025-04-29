class PerformanceAnalysis:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.stack_QoS_parameters = []
        self.stack_Benchmark_QoS_parameters = []
        self.average_parameters = {}
        self.abs_error_parameters = {}
        self.handover_count = 0
        
        
    def store_QoS_parameters(self, QoS_parameters):
        self.stack_QoS_parameters.append(QoS_parameters)
        
    def clean_storaged_QoS(self):
        self.stack_QoS_parameters = []
        
    def store_Benchmark_QoS_parameters(self, QoS_parameters):
        self.stack_Benchmark_QoS_parameters.append(QoS_parameters)
        
    def clean_Benchmark_storaged_QoS(self):
        self.stack_Benchmark_QoS_parameters = []
    
    
    def count_number_of_handovers(self):
        count = 0
        old_protocol = None
        for QoS_parameters in self.stack_QoS_parameters:
            if QoS_parameters['Protocol'] != old_protocol:
                if old_protocol != None:
                    count = count + 1
                old_protocol = QoS_parameters['Protocol']
                    
        self.handover_count = count
        return self.handover_count
    
    
    def calculate_average_QoS_parameters(self, parameters):
        param_dict = {}
        for param in parameters:
            param_dict[param] = []
            
        for QoS_parameters in self.stack_QoS_parameters:
            for QoS_param_key, QoS_param_value in QoS_parameters.items():
                if QoS_param_key in param_dict.keys():
                    param_dict[QoS_param_key].append(QoS_param_value)
        
        self.average_parameters = {key: sum(value) / len(value) for key, value in param_dict.items()}
        return self.average_parameters
    
    
    def calculate_Abs_error_QoS_parameters(self, parameters):        
        QoS_param = {}
        for param in parameters:
            QoS_param[param] = []
        
        Benchmark_param = {}
        for param in parameters:
            Benchmark_param[param] = []
        
        for QoS_parameters in self.stack_QoS_parameters:
            for QoS_param_key, QoS_param_value in QoS_parameters.items():
                if QoS_param_key in QoS_param.keys():
                   QoS_param[QoS_param_key].append(QoS_param_value)
        
        for QoS_parameters in self.stack_Benchmark_QoS_parameters:
            for QoS_param_key, QoS_param_value in QoS_parameters.items():
                if QoS_param_key in Benchmark_param.keys():
                    Benchmark_param[QoS_param_key].append(QoS_param_value)
                    
        for param in parameters:
            self.abs_error_parameters[param] = [abs(QoS_param[param][i] - Benchmark_param[param][i]) for i in range(len(QoS_param[param]))]
        
        return self.abs_error_parameters
    