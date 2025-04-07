class DecisionMakerMethod:

    def __init__(self, method_name):
        self.method_name = method_name
        self.inputs = None
        self.output = None
        
    def send_inputs(self, inputs):
        self.inputs = inputs