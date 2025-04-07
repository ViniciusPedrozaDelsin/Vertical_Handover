from .DecisionMakerMethod import DecisionMakerMethod as DMM

class SPMO_Preference(DMM):
    def __init__(self, method_name, indicator, preference_order):
        super().__init__(method_name)
        self.indicator = indicator
        self.preference_order = preference_order
    
    def makeDecision(self):
        self.output = self.OrderAvailableNetworks()[0]
        return self.output
    
    def OrderAvailableNetworks(self):
        # Sort using the preference order
        output = sorted(self.inputs, key=lambda x: self.preference_order.index(x['Protocol']))
        return output