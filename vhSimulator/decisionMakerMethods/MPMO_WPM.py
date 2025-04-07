from .DecisionMakerMethod import DecisionMakerMethod as DMM
import math

class MPMO_WPM(DMM):
    def __init__(self, method_name, attributes, weights, directions):
        super().__init__(method_name)
        self.attributes = attributes
        self.weights = weights
        self.normalizedWeights = self.normalizeWeights()
        self.directions = directions
        self.normalizedAttributes = None
    
    def makeDecision(self):
        self.normalizedAttributes = self.normalizeAttributes()
        self.output = self.calculate_parameters_cost()
        return self.output
    
    def normalizeWeights(self):
        normalizedWeights = []
        sum = 0
        for weight in self.weights:
            sum = sum + weight
        
        for weight in self.weights:
            normalizedWeights.append(weight/sum)
        
        return normalizedWeights
    
    def normalizeAttributes(self):
        attribute_dict = {}
        normalized_dict = {}
        for attribute in self.attributes:
            attribute_dict[attribute] = []
            normalized_dict[attribute] = []
        
        for input in self.inputs:
            for attribute in self.attributes:
                attribute_dict[attribute].append(input[attribute])
        
        i = 0
        for parameter in attribute_dict:
            max_min_parameter = 0
            j = 0
            for value in attribute_dict[parameter]:
                if self.directions[i] == True:
                    if j == 0:
                        max_min_parameter = value
                    if value > max_min_parameter:
                        max_min_parameter = value
                else:
                    if j == 0:
                        max_min_parameter = value
                    if value < max_min_parameter:
                        max_min_parameter = value
                j = j + 1
                
            for value in attribute_dict[parameter]:
                new_value = value/max_min_parameter
                if new_value > 1: new_value = 1/new_value
                absolute_value = abs(new_value)
                normalized_dict[parameter].append(absolute_value)
            i = i + 1
              
        return normalized_dict
    
    
    def calculate_parameters_cost(self):        
        weighted_data = {key: [val ** self.normalizedWeights[idx] for val in values] for idx, (key, values) in enumerate(self.normalizedAttributes.items())}
        horizontal_products = [math.prod(values) for values in zip(*weighted_data.values())]
        
        # Get max value and its index
        max_value = max(horizontal_products)
        max_index = horizontal_products.index(max_value)
        
        return self.inputs[max_index]