from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np

class NN_RL_RMSE(DMM):
    """
    Optimized and safe version of your NN_RL_RMSE class.
    - Memory entries stored as (state_vector, reward, episode_id)
    - Vectorized training (no per-sample enumerate lookup)
    - Prevents n-step windows that cross simulation boundaries
    - Keeps the same external logic & outputs
    """

    def __init__(self, method_name, attributes, lockin_percentage=None,
                 time_to_trigger=None, model_name=None, simulation_length=50, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes

        # NN RL Variables (kept similar but some defaults as in your last snippet)
        self.epsilon = 0.8
        self.epsilon_min = 0
        self.epsilon_decay = 0.98
        self.batch_size = 32
        self.memory_lenght = 2000
        # memory will hold tuples: (state_vector: tuple, reward: float, episode_id: int)
        self.memory = deque(maxlen=self.memory_lenght)
        self.train_counter = 0

        # episode tracking to avoid cross-simulation windows
        self.simulation_length = simulation_length   # default 50 (you can change)
        self.sim_step_counter = 0
        self.episode_id = 0

        # feature layout: 6 protocol flags + 8 numeric features = 14 inputs
        self.protocol_list = ['WiFi-2.4GHz', 'WiFi-5GHz', 'NB-IoT', 'LoRa-868', 'LTE-4G', 'WiMax']
        # the exact feature order expected by the model / training:
        self.feature_keys = self.protocol_list + ['RSSI', 'SNR', 'Throughput', 'BER', 'FEC', 'PC', 'MC', 'HC']

        if model_name is None:
            self.model = self.modelBuild(14, 1)
        else:
            self.model = tf.keras.models.load_model(model_name)

        # LockIn values
        self.lockin_reference = None
        self.lockin_percentage = lockin_percentage

        # Time to Trigger values
        self.actual_ttt = 0
        self.ttt_reference = None
        self.ttt_active_network = None
        self.time_to_trigger = time_to_trigger

    # ---------------------------
    # High-level decision methods
    # ---------------------------
    def makeDecision(self):
        if self.lockin_percentage is not None:
            self.output = self.makeDecisionLockin()
        elif self.time_to_trigger is not None:
            self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()

        # track step inside simulation; increment episode id if we reached the simulation end
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
        expected_output = self.decisionProcedure()
        self.output = self.check_time_to_trigger(expected_output)
        return self.output

    def decisionProcedure(self):
        normalized_inputs = self.normalizeInputs()
        rmse_rl = self.RMSE_RL(normalized_inputs)
        return rmse_rl

    def resetMemory(self):
        # clearing memory indicates a new simulation run — increment episode id to avoid windows crossing
        self.memory.clear()
        self.episode_id += 1
        self.sim_step_counter = 0

    # ---------------------------
    # Utilities for reward / norm
    # ---------------------------
    def calculateReward(self, normalized_choice):
        reward = (((normalized_choice['RSSI']**2) + (normalized_choice['SNR']**2) + (normalized_choice['BER']**2) +
                   (normalized_choice['FEC']**2) + (normalized_choice['Throughput']**2) + (normalized_choice['PC']**2) +
                   (normalized_choice['MC']**2) + (normalized_choice['HC']**2) + (normalized_choice['Delay']**2) +
                   (normalized_choice['Jitter']**2)) / 10) ** 0.5
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
            normalized_item = item.copy()  # shallow copy is OK since we won't mutate nested structures
            for field in fields:
                min_val = mins[field]
                max_val = maxs[field]
                if max_val == min_val:
                    normalized_item[field] = 0.0
                else:
                    normalized_item[field] = (item[field] - min_val) / (max_val - min_val)
            normalized_data.append(normalized_item)
        return normalized_data

    # ---------------------------
    # Core: RMSE_RL (prediction + memory append)
    # ---------------------------
    def RMSE_RL(self, normalized_inputs):
        """
        Prepare each candidate input for the model (without mutating original dicts),
        predict a scalar score, pick best, compute reward and append vector + reward + episode_id to memory.
        """

        predict_list = []
        prepared_inputs = []   # will hold state_vectors for each candidate (same order as normalized_inputs)

        for item in normalized_inputs:
            # do NOT mutate 'item' in-place. Build new dict/values.
            protocol = item.get('Protocol')

            # encode protocols as 6 flags in the defined protocol_list order
            protocol_flags = [1.0 if p == protocol else 0.0 for p in self.protocol_list]

            # build feature vector in the order expected by the model (protocol flags first, then numeric features)
            numeric_vals = [
                float(item['RSSI']),
                float(item['SNR']),
                float(item['Throughput']),
                float(item['BER']),
                float(item['FEC']),
                float(item['PC']),
                float(item['MC']),
                float(item['HC'])
            ]
            state_vector = tuple(protocol_flags + numeric_vals)  # 14 floats

            # store prepared input vector (we'll use them for predictions and later append one to memory)
            prepared_inputs.append(state_vector)

            # prediction: scalar score from the model (or random exploration)
            prediction = self.modelPrediction(state_vector)
            # ensure prediction is scalar float
            if isinstance(prediction, (list, tuple, np.ndarray)):
                # modelPrediction may return [[val]] or array shape (1,1), handle safely
                pred_val = float(np.array(prediction).flatten()[0])
            else:
                pred_val = float(prediction)
            predict_list.append(pred_val)

        # choose best candidate (max predicted score)
        max_index = int(np.argmax(predict_list))

        # Recover original chosen item so we can compute Delay/Jitter-based reward (without mutating)
        chosen_orig = normalized_inputs[max_index].copy()
        chosen_orig['Delay'] = self.inputs_bkp[max_index]['Delay']
        chosen_orig['Jitter'] = self.inputs_bkp[max_index]['Jitter']

        reward = self.calculateReward(chosen_orig)

        # Append to memory: store state_vector (immutable tuple), reward, episode_id
        chosen_state_vector = prepared_inputs[max_index]
        self.memory.append((chosen_state_vector, float(reward), int(self.episode_id)))

        # occasionally train
        if np.random.rand() > 0.60:
            self.modelTrain()

        # return original (unmodified) decision dict as before
        return self.inputs[max_index]

    # ---------------------------
    # Model helpers
    # ---------------------------
    def modelBuild(self, n_inputs=1, n_outputs=1):
        model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(28, activation='relu'),
            layers.Dense(14, activation='relu'),
            layers.Dense(n_outputs, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss='mse')
        return model

    def modelPrediction(self, model_inputs):
        """
        model_inputs: a sequence of length 14 (protocol flags + 8 features)
        returns a scalar (float) wrapped similarly to your previous code
        """
        if np.random.rand() < self.epsilon:
            # exploration: keep same output shape as model would produce (1,1) -> we'll return scalar though
            return float(np.random.rand())
        else:
            X = np.array([model_inputs], dtype=np.float32)  # shape (1, 14)
            prediction = self.model.predict(X, verbose=0)   # shape (1,1)
            return float(prediction[0, 0])

    def modelTrain(self):
        """
        Vectorized, fast training:
        - build arrays for states, rewards, episode_ids once per call
        - compute windowed sums of length 10 using convolution
        - compute valid window starts where no episode boundary occurs inside the window
        - sample valid starts and train on batches
        """
        mem_len = len(self.memory)
        if mem_len < self.batch_size:
            return

        # determine number of times per call (keeps your train_counter logic)
        if self.train_counter < 100:
            times = 1
        elif 100 <= self.train_counter < 200:
            times = 2
        elif 200 <= self.train_counter < 250:
            times = 3
        else:
            times = 5

        # Convert deque to list once (fast)
        mem_list = list(self.memory)

        # Extract arrays: states, rewards, episode_ids
        # states shape: (mem_len, 14)
        states = np.empty((mem_len, len(self.feature_keys)), dtype=np.float32)
        rewards = np.empty((mem_len,), dtype=np.float32)
        episode_ids = np.empty((mem_len,), dtype=np.int32)

        for i, (state_vec, reward_val, ep_id) in enumerate(mem_list):
            states[i, :] = np.array(state_vec, dtype=np.float32)
            rewards[i] = float(reward_val)
            episode_ids[i] = int(ep_id)

        # sliding-window parameters
        window_len = 10
        if mem_len < window_len:
            return

        # compute windowed sums of rewards (length = mem_len - window_len + 1)
        ones = np.ones(window_len, dtype=np.float32)
        window_sums = np.convolve(rewards, ones, mode='valid')  # shape (mem_len - window_len + 1,)

        # compute change points between consecutive entries
        # change[i] = 1 if episode boundary between i and i+1
        change = (episode_ids[:-1] != episode_ids[1:]).astype(np.int32)  # length mem_len - 1

        # We need to ensure windows of length `window_len` do NOT include any change points.
        # The number of change points inside a window starting at i is:
        # sum(change[i : i + window_len - 1])
        if window_len - 1 > 0:
            window_change_count = np.convolve(change, np.ones(window_len - 1, dtype=np.int32), mode='valid')
        else:
            window_change_count = np.zeros_like(window_sums, dtype=np.int32)

        valid_mask = (window_change_count == 0)  # boolean array length mem_len - window_len + 1
        valid_indices = np.nonzero(valid_mask)[0]  # valid start indices

        if valid_indices.size == 0:
            # No valid windows to train on
            return

        # For each training repeat, sample from valid start indices
        for _ in range(times):
            batch_size = min(self.batch_size, valid_indices.size)
            # sample without replacement from valid indices
            starts = random.sample(list(valid_indices), batch_size)

            X_batch = states[starts]                     # shape (batch_size, 14)
            y_batch = window_sums[starts].astype(np.float32).reshape(-1, 1)  # shape (batch_size, 1)

            # Fit
            self.model.fit(X_batch, y_batch, epochs=1, verbose=0)

            # Decay epsilon per minibatch iteration (keeps original behaviour)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.train_counter += 1

    def saveModel(self):
        self.model.save("RMSE_RL_14inps_32_16.keras")
