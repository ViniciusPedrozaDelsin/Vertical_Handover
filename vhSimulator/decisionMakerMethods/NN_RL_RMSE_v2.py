from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random
import numpy as np

class NN_RL_RMSE(DMM):
    """
    Optimized NN RL using n-step window sums (window_len=10).
    Improvements included:
      - Vectorized training (convert deque -> arrays once)
      - Prevent windows that cross simulation boundaries (episode_id stored)
      - Epsilon exponential decay tuned for long training
      - Huber loss + gradient clipping
      - Warmup steps before training
      - Deterministic training frequency (train every step after warmup)
    External behavior / outputs preserved.
    """

    def __init__(self, method_name, attributes, lockin_percentage=None,
                 time_to_trigger=None, model_name=None, simulation_length=100, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes

        # ----- Hyperparameters (tuned) -----
        self.gamma = 0.95                     # value for future weighting if used
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        # decay_rate computed so epsilon reaches eps_min after decay_episodes
        decay_episodes = 2000.0
        self.epsilon_decay_rate = (self.epsilon_min / 1.0) ** (1.0 / decay_episodes)
        self.batch_size = 64
        self.memory_lenght = 10000
        self.window_len = 10                   # n-step window length (unchanged)
        self.warmup_steps = 1000               # no training until this many entries
        self.train_every_step = True           # deterministic training after warmup
        self.target_update_freq = 1000         # optional: update target network every N training calls
        self.clipnorm = 1.0                    # gradient clipping norm

        # memory holds tuples (state_vector_tuple, reward_float, episode_id_int)
        self.memory = deque(maxlen=self.memory_lenght)
        self.train_counter = 0
        self.train_updates = 0                 # counts actual training updates (for target updates)

        # episode tracking to avoid cross-simulation windows
        self.simulation_length = simulation_length   # default 100 steps per simulation
        self.sim_step_counter = 0
        self.episode_id = 0

        # feature layout: 6 protocol flags + 8 numeric features = 14 inputs
        self.protocol_list = ['WiFi-2.4GHz', 'WiFi-5GHz', 'NB-IoT', 'LoRa-868', 'LTE-4G', 'WiMax']
        self.feature_keys = self.protocol_list + ['RSSI', 'SNR', 'Throughput', 'BER', 'FEC', 'PC', 'MC', 'HC']

        # Model / target model
        if model_name is None:
            self.model = self.modelBuild(14, 1)
        else:
            self.model = tf.keras.models.load_model(model_name)
        # target_model for stability
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # LockIn values
        self.lockin_reference = None
        self.lockin_percentage = lockin_percentage

        # Time to Trigger values
        self.actual_ttt = 0
        self.ttt_reference = None
        self.ttt_active_network = None
        self.time_to_trigger = time_to_trigger

        # make deterministic randomness optional (user can choose to set seeds externally)
        # random.seed(0); np.random.seed(0); tf.random.set_seed(0)

    # ---------------------------
    # High-level decision methods
    # ---------------------------
    def makeDecision(self):
        if self.lockin_percentage is not None:
            self.output = self.makeDecisionLockin()
        elif self.time_to_trigger is not None:
            self.output = self.makeDecisionTimeToTrigger()
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
        # same reward formula as before (negative RMSE-like)
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

    # ---------------------------
    # Core: RMSE_RL (prediction + memory append)
    # ---------------------------
    def RMSE_RL(self, normalized_inputs):
        """
        Prepare each candidate input for the model (without mutating original dicts),
        predict a scalar score, pick best, compute reward and append vector + reward + episode_id to memory.
        """
        predict_list = []
        prepared_inputs = []

        for item in normalized_inputs:
            # encode protocol flags in the defined order
            protocol = item.get('Protocol')
            protocol_flags = [1.0 if p == protocol else 0.0 for p in self.protocol_list]

            numeric_vals = [
                float(item['RSSI']), float(item['SNR']), float(item['Throughput']),
                float(item['BER']), float(item['FEC']), float(item['PC']),
                float(item['MC']), float(item['HC'])
            ]
            state_vector = tuple(protocol_flags + numeric_vals)  # 14 floats
            prepared_inputs.append(state_vector)

            # predict scalar
            prediction = self.modelPrediction(state_vector)
            pred_val = float(np.array(prediction).flatten()[0]) if isinstance(prediction, (list, tuple, np.ndarray)) else float(prediction)
            predict_list.append(pred_val)

        max_index = int(np.argmax(predict_list))

        # compute reward using Delay/Jitter from inputs_bkp (do not mutate originals)
        chosen_orig = normalized_inputs[max_index].copy()
        chosen_orig['Delay'] = self.inputs_bkp[max_index]['Delay']
        chosen_orig['Jitter'] = self.inputs_bkp[max_index]['Jitter']
        reward = self.calculateReward(chosen_orig)

        # append immutable tuple + reward + episode_id
        chosen_state_vector = prepared_inputs[max_index]
        self.memory.append((chosen_state_vector, float(reward), int(self.episode_id)))

        # deterministic training policy: after warmup, train every step (can be adjusted)
        if self.train_every_step and len(self.memory) >= self.warmup_steps:
            self.modelTrain()

        # return original unmodified decision dict
        return self.inputs[max_index]

    # ---------------------------
    # Model helpers
    # ---------------------------
    def modelBuild(self, n_inputs=1, n_outputs=1):
        # Use Huber loss and clipnorm in optimizer for stability
        model = tf.keras.Sequential([
            layers.Input(shape=(n_inputs,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_outputs, activation='linear')
        ])
        # Adam with clipnorm
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=self.clipnorm)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def modelPrediction(self, model_inputs):
        """
        model_inputs: sequence length 14
        returns scalar float
        """
        if np.random.rand() < self.epsilon:
            return float(np.random.rand())
        else:
            X = np.array([model_inputs], dtype=np.float32)
            prediction = self.model.predict(X, verbose=0)
            return float(prediction[0, 0])

    def modelTrain(self):
        """
        Vectorized training pipeline:
         - build arrays (states, rewards, episode_ids)
         - compute window sums via convolution
         - compute valid start indices that do not cross episode boundaries
         - sample batch from valid starts and train
        """
        mem_len = len(self.memory)
        if mem_len < self.batch_size or mem_len < self.window_len:
            return

        # schedule times per training call similar to previous train_counter logic
        if self.train_counter < 100:
            times = 1
        elif 100 <= self.train_counter < 200:
            times = 2
        elif 200 <= self.train_counter < 250:
            times = 3
        else:
            times = 5

        mem_list = list(self.memory)
        # preallocate arrays
        states = np.empty((mem_len, len(self.feature_keys)), dtype=np.float32)
        rewards = np.empty((mem_len,), dtype=np.float32)
        episode_ids = np.empty((mem_len,), dtype=np.int32)

        for i, (state_vec, reward_val, ep_id) in enumerate(mem_list):
            states[i, :] = np.array(state_vec, dtype=np.float32)
            rewards[i] = float(reward_val)
            episode_ids[i] = int(ep_id)

        # sliding window sums (length mem_len - window_len + 1)
        ones = np.ones(self.window_len, dtype=np.float32)
        window_sums = np.convolve(rewards, ones, mode='valid')

        # find change points between consecutive entries (length mem_len - 1)
        change = (episode_ids[:-1] != episode_ids[1:]).astype(np.int32)
        # number of change points inside window starting at i is convolution with window_len-1 ones
        if self.window_len - 1 > 0:
            window_change_count = np.convolve(change, np.ones(self.window_len - 1, dtype=np.int32), mode='valid')
        else:
            window_change_count = np.zeros_like(window_sums, dtype=np.int32)

        valid_mask = (window_change_count == 0)
        valid_indices = np.nonzero(valid_mask)[0]
        if valid_indices.size == 0:
            return

        # For robustness, shuffle valid indices array (we'll sample without replacement)
        valid_list = valid_indices.tolist()

        for _ in range(times):
            batch_size = min(self.batch_size, len(valid_list))
            # sample starts
            starts = random.sample(valid_list, batch_size)

            X_batch = states[starts]
            y_batch = window_sums[starts].astype(np.float32).reshape(-1, 1)

            # train
            self.model.fit(X_batch, y_batch, epochs=1, verbose=0)

            # epsilon decay per minibatch iteration
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay_rate)

            # target network update
            self.train_updates += 1
            if self.train_updates % self.target_update_freq == 0:
                # copy weights for stability
                self.target_model.set_weights(self.model.get_weights())

        self.train_counter += 1

    def saveModel(self):
        self.model.save("RMSE_RL_14inps_64_32.keras")
