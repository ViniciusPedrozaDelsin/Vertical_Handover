from .DecisionMakerMethod import DecisionMakerMethod as DMM

class SPMO_Max_Min_Method(DMM):
    def __init__(self, method_name, indicator, max_value=True):
        super().__init__(method_name)
        self.indicator = indicator
        self.max_value = max_value
    
    def makeDecision(self):
        if self.max_value == True:
            self.output = self.get_maximum_value()
        else:
            self.output = self.get_minimum_value()
        return self.output
    
    def get_maximum_value(self):
        output = None
        max_value = None
        i = 0
        for input in self.inputs:
            if i == 0:
                max_value = input[self.indicator]
                output = input
            else:
                if max_value < input[self.indicator]:
                    max_value = input[self.indicator]
                    output = input
            i = i + 1
        return output
        
    def get_minimum_value(self):
        output = None
        min_value = None
        i = 0
        for input in self.inputs:
            if i == 0:
                min_value = input[self.indicator]
                output = input
            else:
                if min_value > input[self.indicator]:
                    min_value = input[self.indicator]
                    output = input
            i = i + 1
        return output