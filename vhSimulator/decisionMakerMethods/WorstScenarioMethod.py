from .DecisionMakerMethod import DecisionMakerMethod as DMM

class WorstScenarioMethod(DMM):
    def __init__(self, method_name, attributes, directions):
            super().__init__(method_name)
            self.attributes = attributes
            self.directions = directions
        
    def makeDecision(self):
        self.output = self.getWorstValues()
        return self.output
    
    def getWorstValues(self):
        # Sort using the preference order
        values_dict = {}
        
        for att in self.attributes:
            values_dict[att] = []
            
        for input in self.inputs:
            for att in self.attributes:
                values_dict[att].append(input[att])
        
        best_values = []
        i = 0
        for att in self.attributes:
            if self.directions[i] == 0:
                best_values.append(max(values_dict[att]))
            else:
                best_values.append(min(values_dict[att]))
            i = i + 1
        
        j = 0
        input_benchmark = {'Network': 'WorstScenario', 'Status': 'Online', 'Protocol': 'WorstScenario'}
        for att in self.attributes:
            input_benchmark[att] = best_values[j]
            j = j + 1
        
        output = input_benchmark
        return output