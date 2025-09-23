from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np

class NN_RL_RMSE(DMM):
    def __init__(self, method_name, attributes, lockin_percentage=None, time_to_trigger=None, simulation_length=100, model_name=None, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes
        
        # NN RL Variables
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.window_size = 5
        self.memory_lenght = 8192
        self.memory = deque(maxlen=self.memory_lenght)
        self.mem_warmup_steps = 1024
        self.train_counter = 0
        if model_name == None:
            self.model = self.modelBuild(14, 1)
        else:
            self.model = tf.keras.models.load_model(model_name)

        '''# --- Target network (same architecture) ---
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # update frequency
        self.target_update_freq = 1000'''
        

        # Episode tracking to avoid cross-simulation windows
        self.simulation_length = simulation_length   # default 50 (you can change)
        self.sim_step_counter = 0
        self.episode_id = 0

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
            self.output = self.makeDecisionLockin()
        elif self.time_to_trigger != None:
            self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()
        
        # Track step inside simulation; increment episode id if we reached the end of the simulation
        self.sim_step_counter += 1
        if self.sim_step_counter >= self.simulation_length:
            self.episode_id += 1
            self.sim_step_counter = 0

        self.output = self.return_output()
        self.old_decision = self.output['Network']
        return self.output
    
    
    def makeDecisionLockin(self):
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
        normalized_inputs = self.normalizeInputs()
        rmse_rl = self.RMSE_RL(normalized_inputs)
        return rmse_rl
    
    def resetMemory(self):
        self.memory.clear()
    
    def calculateReward(self, normalized_choice):
        reward = (((normalized_choice['RSSI']**2) + (normalized_choice['SNR']**2) + (normalized_choice['BER']**2) + (normalized_choice['FEC']**2) + (normalized_choice['Throughput']**2) + (normalized_choice['PC']**2) + (normalized_choice['MC']**2) + (normalized_choice['HC']**2) + (normalized_choice['Delay']**2) + (normalized_choice['Jitter']**2))/10)**(1/2)
        return -reward
        
    def normalizeInputs(self):
        # Fields to Normalize
        fields = ['RSSI', 'SNR', 'BER', 'FEC', 'Throughput', 'PC', 'MC', 'HC']

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
        
    
    def RMSE_RL(self, normalized_inputs):
        
        # Encode Network Protocol
        protocol_list = ['WiFi-2.4GHz', 'WiFi-5GHz', 'NB-IoT', 'LoRa-868', 'LTE-4G', 'WiMax']
            
        predict_list = []
        for inp in normalized_inputs:
            
            # Deleting useless informations
            del inp['Network']
            del inp['Status']
            del inp['Delay']
            del inp['Jitter']
            del inp['Distance']
            
            # Encoding Networks Protocol
            protocol_encoded_dict = {}
            for protocol_type in protocol_list:
                if protocol_type == inp['Protocol']:
                    protocol_encoded_dict[protocol_type] = 1
                else:
                    protocol_encoded_dict[protocol_type] = 0
            del inp['Protocol']
            
           
            # Correct Dictionary Order
            reordered = {
                'RSSI': inp['RSSI'],
                'SNR': inp['SNR'],
                'Throughput': inp['Throughput'],
                'BER': inp['BER'],
                'FEC': inp['FEC'],
                'PC': inp['PC'],
                'MC': inp['MC'],
                'HC': inp['HC']
            }
            
            new_inp = protocol_encoded_dict | reordered
            
            params = []
            for key, value in new_inp.items():
                params.append(value)
            
            prediction = self.modelPrediction(params)
            predict_list.append(prediction)
        
        max_index = np.argmax(predict_list)
        
        new_inp['Delay'] = self.inputs_bkp[max_index]['Delay']
        new_inp['Jitter'] = self.inputs_bkp[max_index]['Jitter']
        
        reward = self.calculateReward(new_inp)
        
        del new_inp['Delay']
        del new_inp['Jitter']
        
        self.memory.append((new_inp, reward))
        if np.random.rand() > 0.75:
            self.modelTrain()
        
        return self.inputs[max_index]
    
    def modelBuild(self, n_inputs=1, n_outputs=1):
        model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(n_outputs, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss='mse')
        return model
    
    def modelPrediction(self, model_inputs):
        if np.random.rand() < self.epsilon:
            prediction = [[np.random.rand()]]
        else:
            X = np.array([model_inputs])
            prediction = self.model.predict(X, verbose=0)
        return prediction
    
    def modelTrain(self):
        # Return if Memory < Batch_Size
        if len(self.memory) < (self.batch_size + self.window_size):
            return
        
        # Return util memory warmup
        if len(self.memory) < self.mem_warmup_steps:
            return

        if self.train_counter < 100:
            times = 1
        elif self.train_counter >= 100 and self.train_counter < 200:
            times = 1
        elif self.train_counter >= 200 and self.train_counter < 250:
            times = 1
        else:
            times = 2
            
        for _ in range(times):
            minibatch = random.sample(list(self.memory)[:-self.window_size], self.batch_size)
            
            X = []
            y = []
            for sample_choice in minibatch:
                target = 0
                index = next((i for i, sc in enumerate(self.memory) if sc == sample_choice), None)
                if (index % self.simulation_length) <= (self.simulation_length - self.window_size):
                    for i in range(self.window_size):
                        target += (self.memory[index+i][1]) * (self.gamma**i)
                    X.append(list(self.memory[index][0].values()))
                    y.append(target)
            
            self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        self.train_counter += 1

        '''# update target model every N updates
        if self.train_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())'''
    
    def saveModel(self):
        #self.model.save(self.model_name)
        self.model.save("RMSE_RL_14inps_32_16.keras")