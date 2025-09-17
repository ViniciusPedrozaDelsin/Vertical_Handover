from .DecisionMakerMethod import DecisionMakerMethod as DMM
import numpy as np

class MPMO_TOPSIS(DMM):
    def __init__(self, method_name, attributes, weights, directions, hysterese_percentage=None, lockin_percentage=None, time_to_trigger=None, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes
        self.weights = weights
        self.normalizedWeights = self.normalizeWeights()
        self.directions = directions
        self.attributes_matrix = None
        self.normalized_attributes_matrix = None
        
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
            self.output = self.makeDecisionLockIn()
        elif self.time_to_trigger != None:
            self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()
            
        self.output = self.return_output()
        self.old_decision = self.output['Network']
        return self.output
    
    
    def makeDecisionLockIn(self):
        check_lockin_reference = self.check_lockin_reference()
        if check_lockin_reference[0]:
            self.output = self.decisionProcedure()
            self.lockin_reference = self.output
        else:
            self.output = check_lockin_reference[1]
        return self.output
        
    def makeDecisionTimeToTrigger(self):
        # Generate expected output
        expected_output = self.decisionProcedure()
        self.output = self.check_time_to_trigger(expected_output)
        return self.output
    
    def decisionProcedure(self):
        self.attributes_matrix = self.get_Attributes_Matrix()
        self.normalized_attributes_matrix = self.NormalizeAttributesMatrix()
        attributes_weights_matrix = self.get_matrix_attributes_weights()
        output = self.calculateSolution(attributes_weights_matrix)
        return output
    
    def normalizeWeights(self):
        normalizedWeights = []
        sum = 0
        for weight in self.weights:
            sum = sum + weight
        
        for weight in self.weights:
            normalizedWeights.append(weight/sum)
        
        return np.array(normalizedWeights, dtype=float)
    
    def get_Attributes_Matrix(self):
        attributes_matrix = []
        for input in self.inputs:
            input_list = []
            for attribute in self.attributes:
                input_list.append(input[attribute])
            attributes_matrix.append(input_list)
        return np.array(attributes_matrix, dtype=float)
    
    
    def NormalizeAttributesMatrix(self):
        dem = np.sqrt((self.attributes_matrix ** 2).sum(axis=0))
        
        # Replace zeros with ones
        dem[dem == 0] = 1
        
        norm_matrix = self.attributes_matrix / dem
        return norm_matrix
    
    
    def get_matrix_attributes_weights(self):
        attributes_matrix_multiplied_by_weights = self.normalized_attributes_matrix * self.normalizedWeights
        return attributes_matrix_multiplied_by_weights
        
    def calculateSolution(self, matrix):
        # Determine ideal and negative-ideal solutions
        i = 0
        a_plus_list = []
        a_minus_list = []
        for direction in self.directions:
            if direction == True:
                a_plus = np.max(matrix[:,i], axis=0)
                a_minus = np.min(matrix[:,i], axis=0)
            else:
                a_plus = np.min(matrix[:,i], axis=0)
                a_minus = np.max(matrix[:,i], axis=0)
            a_plus_list.append(a_plus)
            a_minus_list.append(a_minus)
            i = i + 1
        a_plus_matrix = np.array(a_plus_list, dtype=float)
        a_minus_matrix = np.array(a_minus_list, dtype=float)
        
        # Calculate Euclidean distances
        dist_ideal = np.sqrt(((matrix - a_plus_matrix) ** 2).sum(axis=1))
        dist_negative_ideal = np.sqrt(((matrix - a_minus_matrix) ** 2).sum(axis=1))

        # Calculate relative closeness to ideal solution
        if np.all(dist_ideal == 0) and np.all(dist_negative_ideal == 0):
            closeness_coefficient = [1]
        else:
            closeness_coefficient = dist_negative_ideal / (dist_ideal + dist_negative_ideal)
        
        # Save data into .CSV
        if self.file_to_save != None:
            self.saveData(closeness_coefficient)
        
        # Rank alternatives (higher is better)
        # Get the index of the max value
        max_index = np.argmax(closeness_coefficient)
        output = self.inputs[max_index]
        
        # Hysteresis
        if self.hysterese_percentage != None:
            networks = [net['Network'] for net in self.inputs]
            if self.old_decision in networks:
                if self.old_decision != output['Network']:
                    if closeness_coefficient[networks.index(self.old_decision)] * (1 + self.hysterese_percentage) >=  np.max(closeness_coefficient):
                        output = self.inputs[networks.index(self.old_decision)]
        
        return output