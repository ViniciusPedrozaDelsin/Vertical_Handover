from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
import numpy as np

class NN_TOPSIS(DMM):
    def __init__(self, method_name, attributes, hysterese_percentage=None, time_to_trigger=None, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes
        self.model = tf.keras.models.load_model("TOPSIS_NN_7_6_2.keras")
        
        # Hysteresis values
        self.hysteresis_reference = None
        self.hysterese_percentage = hysterese_percentage
        
        # Time to Trigger values
        self.actual_ttt = 0
        self.ttt_reference = None
        self.ttt_active_network = None
        self.time_to_trigger = time_to_trigger
    
    def makeDecision(self):
        if self.hysterese_percentage != None:
            self.output = self.makeDecisionHysteresis()
        elif self.time_to_trigger != None:
            self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()
        return self.output
    
    
    def makeDecisionHysteresis(self):
        check_hysteresis_reference = self.check_hysteresis_reference()
        if check_hysteresis_reference[0]:
            self.output = self.decisionProcedure()
            self.hysteresis_reference = self.output
        else:
            self.output = check_hysteresis_reference[1]
        return self.output
        
    def makeDecisionTimeToTrigger(self):
        # Generate expected output
        expected_output = self.decisionProcedure()
        self.output = self.check_time_to_trigger(expected_output)
        return self.output
    
    def decisionProcedure(self):
        normalized_inputs = self.normalizeInputs()
        simulate_topsis = self.simulateTopsis(normalized_inputs)
        return simulate_topsis
        
    def normalizeInputs(self):
        # Fields to Normalize
        fields = ['RSSI', 'SNR', 'BER', 'FEC', 'Throughput', 'PC', 'MC']

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
        return normalized_data
        
    
    def simulateTopsis(self, normalized_inputs):
        
        predict_list = []
        for inp in normalized_inputs:
        
            # Deleting useless informations
            del inp['Network']
            del inp['Status']
            del inp['Distance']
            del inp['Protocol']
           
            # Correct Dictionary Order
            reordered = {
                'RSSI': inp['RSSI'],
                'SNR': inp['SNR'],
                'Throughput': inp['Throughput'],
                'BER': inp['BER'],
                'FEC': inp['FEC'],
                'PC': inp['PC'],
                'MC': inp['MC']
            }
            
            inp = reordered
            
            params = []
            for key, value in inp.items():
                params.append(value)
            
            X = np.array([params])
            
            prediction = self.model.predict(X, verbose=0)
            
            predict_list.append(prediction)
        
        max_index = np.argmax(predict_list)
        
        return self.inputs[max_index]