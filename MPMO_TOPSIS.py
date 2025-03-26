from DecisionMakerMethod import DecisionMakerMethod as DMM
import numpy as np

class MPMO_TOPSIS(DMM):
    def __init__(self, method_name, attributes, weights, directions):
        super().__init__(method_name)
        self.attributes = attributes
        self.weights = weights
        self.normalizedWeights = self.normalizeWeights()
        self.directions = directions
        self.attributes_matrix = None
        self.normalized_attributes_matrix = None
    
    def makeDecision(self):
        self.attributes_matrix = self.get_Attributes_Matrix()
        self.normalized_attributes_matrix = self.NormalizeAttributesMatrix()
        attributes_weights_matrix = self.get_matrix_attributes_weights()
        self.output = self.calculateSolution(attributes_weights_matrix)
        return self.output
    
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
        norm_matrix = self.attributes_matrix / np.sqrt((self.attributes_matrix ** 2).sum(axis=0))
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
        closeness_coefficient = dist_negative_ideal / (dist_ideal + dist_negative_ideal)
        
        # Rank alternatives (higher is better)
        # Get the index of the max value
        max_index = np.argmax(closeness_coefficient)
        
        return self.inputs[max_index]