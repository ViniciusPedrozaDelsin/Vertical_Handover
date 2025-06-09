from .DecisionMakerMethod import DecisionMakerMethod as DMM
import copy

class MPMO_RMSE(DMM):
    def __init__(self, method_name, attributes, weights, directions, hysterese_percentage=None, time_to_trigger=None):
        super().__init__(method_name)
        self.attributes = attributes
        self.weights = weights
        self.directions = directions
        
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
            self.output = self.makeDecisionTimeToTrigger()
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
        output = self.calculateRMSE()
        return output
    
    def getBenchmarkParameters(self):
        attribute_dict = {}
        for attribute in self.attributes:
            attribute_dict[attribute] = []
        
        for input in self.inputs:
            for attribute in self.attributes:
                attribute_dict[attribute].append(copy.deepcopy(input[attribute]))
        
        j = 0
        benchmark_dict = {}
        for k, v in attribute_dict.items():
            if self.directions[j] == 0:
                benchmark_dict[k] = min(v)
            else:
                benchmark_dict[k] = max(v)
            j = j + 1

        return benchmark_dict
    
    def normalizeInputs(self):
        pass
    
    def calculateRMSE(self):
        benchmark_parameters = self.getBenchmarkParameters()
        # Get the Absolute Value
        abs_inputs = []
        for input in self.inputs:
            abs_input_dict = copy.deepcopy(input)
            for attribute in self.attributes:
                abs_val = abs(input[attribute] - benchmark_parameters[attribute])
                abs_input_dict[attribute] = abs_val
            abs_inputs.append(abs_input_dict)
        # Normalizing Values
        parm_dict = {}
        for attribute in self.attributes:
            parm_dict[attribute] = []
            for abs_input in abs_inputs:
                parm_dict[attribute].append(abs_input[attribute])
        e = 0.000000001
        for k, v in parm_dict.items():
            new_value = [(x-min(v))/((max(v)-min(v)) + e) for x in v]
            parm_dict[k] = new_value
        # Transpose the dictionary
        transposed = {
            str(i): [parm_dict[key][i] for key in parm_dict]
            for i in range(len(next(iter(parm_dict.values()))))
        }
        rmse_list = []
        n_parm = 0
        for k, v in transposed.items():
            sum_value = 0
            i = 0
            for value in v:
                sum_value = sum_value + (self.weights[i] * (value**2))
                i = i + 1
            rmse_list.append(sum_value)
            n_parm = len(v)
        final_rmse_list = [(x/n_parm)**(1/2) for x in rmse_list]
        
        min_index = final_rmse_list.index(min(final_rmse_list))
        output = self.inputs[min_index]
        return output