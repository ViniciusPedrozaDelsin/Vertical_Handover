from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from collections import deque
import random
import numpy as np
import pandas as pd
import shap

class NN_RL_RMSE(DMM):
    def __init__(self, method_name, attributes, lockin_percentage=None, time_to_trigger=None, simulation_length=100, model_name=None, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes

        # NN RL Variables
        self.gamma = 0.95
        self.epsilon = 0.00
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.9975
        self.batch_size = 32
        self.window_size = 10
        self.memory_lenght = 8192
        self.memory = deque(maxlen=self.memory_lenght)
        self.mem_warmup_steps = 2048
        self.train_counter = 0
        self.memory_velocity = {}
        self.memory_velocity_len = 4
        self.model_name = model_name
        if self.model_name == None:
            self.model = self.modelBuild(8, 1, self.memory_velocity_len-1, 1)
        else:
            self.model = tf.keras.models.load_model(model_name)
        
        # SHAP Analysis
        self.shap_analysis = False
        if self.shap_analysis:
            self.memory_shap = deque(maxlen=500)
            self.memory_shap_warmup = 500
            self.shap_chance = 0.025
            
        # Training Analysis
        self.train_analysis = True
        if self.train_analysis:
            self.training_losses = []
            self.reward_history = []
            self.reward_step_counter = 0

        # Episode tracking to avoid cross-simulation windows
        self.simulation_length = simulation_length 
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

    def resetMemoryVelocity(self):
        self.memory_velocity.clear()
    
    def resetMemoryShap(self):
        self.memory_shap.clear()


    def calculateReward(self, normalized_choice):
        reward = (((normalized_choice['RSSI']**2) + (normalized_choice['SNR']**2) + (normalized_choice['BER']**2) + (normalized_choice['FEC']**2) + (normalized_choice['Throughput']**2) + (normalized_choice['PC']**2) + (normalized_choice['MC']**2) + (normalized_choice['HC']**2) + (normalized_choice['Delay']**2) + (normalized_choice['Jitter']**2))/10)**(1/2)
        #print("========= NN RL RMSE ========")
        #print(f"- NN RL RMSE Reward {reward}")
        #print("=============================")
        return -reward

    def getInputsBkpNormalize(self, param, index):
        max_value = None
        min_value = None
        target_value = None

        i = 0
        for input_value in self.inputs_bkp:

            if i == index:
                target_value = input_value[param]

            if i == 0:
                max_value = input_value[param]
                min_value = input_value[param]

            if input_value[param] > max_value:
                max_value = input_value[param]

            if input_value[param] < min_value:
                min_value = input_value[param]

            i += 1

        if max_value == min_value:
            value_normalized = 0
        else:
            value_normalized = (target_value - min_value) / (max_value - min_value)

        #print(f"Param: {param} | Value: {value_normalized}")
        return value_normalized


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
                    if field == 'RSSI' or field == 'SNR' or field == 'FEC' or field == 'Throughput':
                        normalized_item[field] = (max_val - item[field]) / (max_val - min_val)
                    elif field == 'BER' or field == 'PC' or field == 'MC' or field == 'HC':
                        normalized_item[field] = (item[field] - min_val) / (max_val - min_val)
            normalized_data.append(normalized_item)
        return normalized_data


    def RMSE_RL(self, normalized_inputs):

        # Encode Network Protocol
        protocol_list_dict = {'WiFi-2.4GHz': 0, 'WiFi-5GHz': 1, 'NB-IoT': 2, 'LoRa-868': 3, 'LTE-4G': 4, '5G-N78': 5, '5G-N28': 6}
        inpt_list = []

        predict_list = []
        for inp in normalized_inputs:

            # Deleting useless informations
            del inp['Status']
            del inp['Delay']
            del inp['Jitter']

            if inp['Network'] not in self.memory_velocity:
                self.memory_velocity[inp['Network']] = deque(maxlen=self.memory_velocity_len)

            # Add Distance into memory
            self.memory_velocity[inp['Network']].append(inp['Distance'])

            # Calculate Velocity
            if len(self.memory_velocity[inp['Network']]) == self.memory_velocity_len:
                #velocity_dict = {f'VEL_{i+1}': ((self.memory_velocity[inp['Network']][i+1] - self.memory_velocity[inp['Network']][i]) / self.memory_velocity[inp['Network']][i+1]) * (100) for i in range(len(self.memory_velocity[inp['Network']])-1)}
                velocity_dict = {f'VEL_{i+1}': (self.memory_velocity[inp['Network']][i+1] - self.memory_velocity[inp['Network']][i]) * (10) for i in range(len(self.memory_velocity[inp['Network']])-1)}
            else:
                velocity_dict = {'VEL_1': 0, 'VEL_2': 0, 'VEL_3': 0}
            #print("==========================================")
            #print(self.memory_velocity)
            #print(velocity_dict)
            #print("==========================================")

            del inp['Distance']
            del inp['Network']

            # Encoding Networks Protocol - Embedding Vector
            protocol_encoded_dict = {}
            for prot, number in protocol_list_dict.items():
                if inp['Protocol'] == prot:
                    protocol_encoded_dict['Protocol'] = number
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


            new_inp = protocol_encoded_dict | velocity_dict
            new_inp = new_inp | reordered
            inpt_list.append(new_inp)
            
            params = []
            for key, value in new_inp.items():
                params.append(value)
            inpt_list.append(new_inp)
            predict_list.append(params)

        # Batch predict all networks in ONE call
        all_protocols  = np.array([p[0]                          for p in predict_list])
        all_velocities = np.array([p[1:self.memory_velocity_len] for p in predict_list])
        all_params     = np.array([p[self.memory_velocity_len:]  for p in predict_list])
        
        predict_list = self.modelPrediction(all_protocols, all_velocities, all_params)

        #print("====================")
        #print(predict_list)
        max_index = np.argmax(predict_list)
        #print(normalized_inputs)
        #print(max_index)
        #print("-- INPUT LIST --")
        #print(inpt_list)
        #print("====================")

        inpt_list[max_index]['Delay'] = self.getInputsBkpNormalize('Delay', max_index)
        inpt_list[max_index]['Jitter'] = self.getInputsBkpNormalize('Jitter', max_index)

        reward = self.calculateReward(inpt_list[max_index])
        if self.train_analysis:
            self.reward_history.append({
                'step': self.reward_step_counter,
                'reward': reward
            })
            self.reward_step_counter += 1

        del inpt_list[max_index]['Delay']
        del inpt_list[max_index]['Jitter']

        self.memory.append((inpt_list[max_index], reward))

        if np.random.rand() < 0.01:
            self.modelTrain()

        return self.inputs[max_index]

    def modelBuild(self, n_inputs=1, n_protocols=1, n_velocities=1, n_outputs=1):

        protocol = Input(shape=(n_protocols,), dtype='int32', name='protocol')
        velocities = Input(shape=(n_velocities,), dtype='float32', name='velocities')
        inputs = Input(shape=(n_inputs,), dtype='float32', name='inputs')

        # Embedding layer
        emb = layers.Embedding(input_dim=7, output_dim=4, embeddings_initializer='glorot_uniform', name='protocol_embedding')(protocol)
        emb = layers.Flatten()(emb)

        vel_seq = layers.Reshape((n_velocities, 1))(velocities)
        lstm_out = layers.LSTM(8, activation='tanh')(vel_seq)
        lstm_scalar = layers.Dense(1, activation='tanh', name='lstm_scalar')(lstm_out)

        # Combine Inputs
        x = layers.Concatenate()([emb, lstm_scalar, inputs])

        # MLP (32 -> 64 -> 32 -> 16)
        x = layers.Dense(32, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(64, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(32, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Dense(16, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        out = layers.Dense(1, activation='linear', name='q_out')(x)

        model = Model(inputs=[protocol, velocities, inputs], outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model
    
    def Execute_SHAP_Analysis(self, inpts_protocol, inpts_velocities, inpts_model):
        feature_vector = np.concatenate([[inpts_protocol], np.array(inpts_velocities).flatten(), np.array(inpts_model).flatten()])
        self.memory_shap.append(feature_vector)
        if len(self.memory_shap) >= self.memory_shap_warmup:
            shap_sample = np.array(list(self.memory_shap)[:500])  # shape (30, 12)
            feature_names = ['Protocol', 'VEL_1', 'VEL_2', 'VEL_3', 'RSSI', 'SNR', 'Throughput', 'BER', 'FEC', 'PC', 'MC', 'HC']

            # Define a wrapper for SHAP
            def shap_model(X_input):
                protocol_col = X_input[:, 0:1]
                velocity_cols = X_input[:, 1:4]
                param_cols = X_input[:, 4:]
                return self.model.predict([protocol_col, velocity_cols, param_cols], verbose=0).flatten()

            explainer = shap.Explainer(shap_model, shap_sample)
            shap_values = explainer(shap_sample)
            shap.summary_plot(shap_values.values, shap_sample, feature_names=feature_names)
            shap.summary_plot(shap_values.values, shap_sample, feature_names=feature_names, plot_type="bar")
            self.resetMemoryShap()
    
    def getLearningCurves(self):
        return {
            'losses': pd.DataFrame(self.training_losses),
            'rewards': pd.DataFrame(self.reward_history)
        }
    
    def modelPrediction(self, protocol_inputs, velocity_inputs, model_inputs):
        if np.random.rand() < self.epsilon:
            prediction = []
            for _ in range(len(protocol_inputs)):
                prediction.append([np.random.rand()])
        else:
            X = [protocol_inputs, velocity_inputs, model_inputs]
            prediction = self.model.predict(X, verbose=0)
            if self.shap_analysis and np.random.rand() < self.shap_chance:
                for i in range(len(protocol_inputs)):
                    self.Execute_SHAP_Analysis(protocol_inputs[i], velocity_inputs[i], model_inputs[i])
        return prediction

    def modelTrain(self):
        # Return if Memory < Batch_Size
        if len(self.memory) < (self.batch_size + self.window_size):
            return

        # Return util memory warmup
        if len(self.memory) < self.mem_warmup_steps:
            return

        if self.train_counter < 150:
            times = 1
        elif self.train_counter >= 150 and self.train_counter < 200:
            times = 1
        elif self.train_counter >= 200 and self.train_counter < 500:
            times = 1
        else:
            times = 1

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

            # List of the Protocols
            protocol_num_list = [sublist[0] for sublist in X]

            # List of the Velocities
            velocities_list = [sublist[1:self.memory_velocity_len] for sublist in X]

            # List of the Parameters
            parameters_list = [sublist[self.memory_velocity_len:] for sublist in X]

            # Set the learning rate to 0.00001
            if self.train_counter > 0:
                self.model.optimizer.learning_rate.assign(3e-5)
            
            history = self.model.fit([np.array(protocol_num_list), np.array(velocities_list), np.array(parameters_list)], np.array(y), epochs=1, verbose=0)
            if self.train_analysis:
                self.training_losses.append({
                    'train_step': self.train_counter,
                    'loss': history.history['loss'][0]
                })
                
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.train_counter += 1


    def saveModel(self):
        if self.model_name != None:
            self.model.save(self.model_name)
        else:
            self.model.save("RMSE_RL.keras")