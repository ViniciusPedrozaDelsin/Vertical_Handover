from .DecisionMakerMethod import DecisionMakerMethod as DMM
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from collections import deque
import random
import numpy as np

class NN_RL_RMSE(DMM):
    """
    Improved NN RL decision maker:
      - Vectorized batch predictions (fast)
      - Replay memory stored as (state_tuple, reward, episode_id)
      - n-step bootstrapped targets using a target network
      - Huber loss, gradient clipping, BatchNorm and Dropout
      - Protocol-aware running stats for Delay/Jitter appended to inputs
      - Robust episode boundary handling
    """

    def __init__(self, method_name, attributes,
                 lockin_percentage=None, time_to_trigger=None,
                 simulation_length=100, model_name=None, **kwargs):
        super().__init__(method_name, **kwargs)
        self.attributes = attributes

        # -----------------------
        # Hyperparameters
        # -----------------------
        self.protocol_list = ['WiFi-2.4GHz', 'WiFi-5GHz', 'NB-IoT', 'LoRa-868', 'LTE-4G', 'WiMax']
        self.n_protocols = len(self.protocol_list)

        # Input layout:
        #  - protocol one-hot flags (6)
        #  - numeric features (8)
        #  - appended: per-protocol avg Delay and avg Jitter (2)
        # => total inputs = 6 + 8 + 2 = 16
        self.input_size = self.n_protocols + 8 + 2

        # RL hyperparams
        self.gamma = 0.90                 # discounting for bootstrapping
        self.window_size = 7              # n-step (7-step) window
        self.batch_size = 64
        self.memory_length = 16384        # replay memory capacity
        self.mem_warmup_steps = 1024      # don't train before this many samples
        self.batch_train_repeats = 1      # number of minibatches per modelTrain() call (tunable)

        # epsilon schedule (exploration)
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = 0.9995      # multiplicative decay per minibatch

        # optimizer / model stability
        self.learning_rate = 3e-4
        self.clipnorm = 1.0

        # target network update
        self.target_update_freq = 1000    # hard sync every this many training updates
        self.soft_update_tau = None       # if set to e.g. 0.01, use soft updates instead of hard sync
        self.train_updates = 0

        # memory and counters
        self.memory = deque(maxlen=self.memory_length)  # holds tuples (state_tuple, reward, episode_id)
        self.train_counter = 0

        # episode tracking
        self.simulation_length = simulation_length
        self.sim_step_counter = 0
        self.episode_id = 0

        # Keep running per-protocol statistics for Delay & Jitter (so agent can infer hidden info)
        # store as {'protocol_label': {'count': int, 'avg_delay': float, 'avg_jitter': float}}
        self.protocol_stats = {p: {'count': 0, 'avg_delay': 0.0, 'avg_jitter': 0.0} for p in self.protocol_list}

        # build / load model and target model
        if model_name is None:
            self.model = self.modelBuild(self.input_size, 1)
        else:
            self.model = tf.keras.models.load_model(model_name)

        # create target model and copy weights initially
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

        # lockin / TTT placeholders
        self.lockin_reference = None
        self.lockin_percentage = lockin_percentage
        self.actual_ttt = 0
        self.ttt_reference = None
        self.ttt_active_network = None
        self.time_to_trigger = time_to_trigger

    # ---------------------------
    # High level decision flow
    # ---------------------------
    def makeDecision(self):
        if self.lockin_percentage is not None:
            self.output = self.makeDecisionLockin()
        elif self.time_to_trigger is not None:
            self.output = self.makeDecisionTimeToTrigger()
        else:
            self.output = self.decisionProcedure()

        # episode/step bookkeeping
        self.sim_step_counter += 1
        if self.sim_step_counter >= self.simulation_length:
            self.episode_id += 1
            self.sim_step_counter = 0

        self.output = self.return_output()
        self.old_decision = self.output['Network']
        return self.output

    def makeDecisionLockin(self):
        check = self.check_lockin_reference()
        if check[0]:
            self.output = self.decisionProcedure()
            self.lockin_reference = self.output
        else:
            self.output = check[1]
        return self.output

    def makeDecisionTimeToTrigger(self):
        expected_output = self.decisionProcedure()
        self.output = self.check_time_to_trigger(expected_output)
        return self.output

    def decisionProcedure(self):
        normalized_inputs = self.normalizeInputs()
        return self.RMSE_RL(normalized_inputs)

    def resetMemory(self):
        self.memory.clear()
        # optionally reset protocol stats as new run starts
        for p in self.protocol_stats:
            self.protocol_stats[p] = {'count': 0, 'avg_delay': 0.0, 'avg_jitter': 0.0}

    # ---------------------------
    # Reward and normalization
    # ---------------------------
    def calculateReward(self, normalized_choice):
        # root-mean-square-like across the 10 metrics (including Delay & Jitter)
        # normalized_choice must contain Delay and Jitter (we add them only for chosen sample)
        reward = (((normalized_choice['RSSI']**2) + (normalized_choice['SNR']**2) +
                   (normalized_choice['BER']**2) + (normalized_choice['FEC']**2) +
                   (normalized_choice['Throughput']**2) + (normalized_choice['PC']**2) +
                   (normalized_choice['MC']**2) + (normalized_choice['HC']**2) +
                   (normalized_choice['Delay']**2) + (normalized_choice['Jitter']**2)) / 10.0) ** 0.5
        # return negative because lower RMSE is better
        return -float(reward)

    def normalizeInputs(self):
        # Fields to Normalize (numeric)
        fields = ['RSSI', 'SNR', 'BER', 'FEC', 'Throughput', 'PC', 'MC', 'HC']

        # compute min/max across self.inputs (current candidates)
        mins = {field: min(d[field] for d in self.inputs) for field in fields}
        maxs = {field: max(d[field] for d in self.inputs) for field in fields}

        normalized = []
        for item in self.inputs:
            ni = item.copy()
            for f in fields:
                lo = mins[f]; hi = maxs[f]
                if hi == lo:
                    ni[f] = 0.0
                else:
                    ni[f] = (item[f] - lo) / (hi - lo)
            normalized.append(ni)
        return normalized

    # ---------------------------
    # Core decision: batch predict + memory append
    # ---------------------------
    def RMSE_RL(self, normalized_inputs):
        """
        - prepare a batch X of shape (n_candidates, input_size)
        - predict all at once with self.model
        - choose argmax score
        - compute reward for chosen candidate (inject Delay/Jitter from inputs_bkp)
        - update protocol running stats (Delay/Jitter)
        - append (state_tuple, reward, episode_id) to memory
        - trigger modelTrain() occasionally
        """
        protocol_flags = self.protocol_list

        prepared_inputs = []   # each is a list of floats length input_size
        # keep mapping idx->original normalized dict for reward computation
        for item in normalized_inputs:
            # protocol one-hot
            p = item.get('Protocol')
            flags = [1.0 if proto == p else 0.0 for proto in protocol_flags]

            numeric_vals = [
                float(item['RSSI']), float(item['SNR']), float(item['Throughput']),
                float(item['BER']), float(item['FEC']), float(item['PC']),
                float(item['MC']), float(item['HC'])
            ]

            # append running per-protocol avg Delay & Jitter as extra inputs:
            stats = self.protocol_stats.get(p, {'avg_delay': 0.0, 'avg_jitter': 0.0})
            avg_delay = float(stats['avg_delay'])
            avg_jitter = float(stats['avg_jitter'])

            vec = flags + numeric_vals + [avg_delay, avg_jitter]
            prepared_inputs.append(vec)

        X = np.array(prepared_inputs, dtype=np.float32)  # shape (n_candidates, input_size)

        # epsilon-greedy at candidate level (we assign random scores to all candidates when exploring)
        if np.random.rand() < self.epsilon:
            scores = np.random.rand(X.shape[0]).astype(np.float32)
        else:
            preds = self.model.predict(X, verbose=0).reshape(-1)   # shape (n_candidates,)
            scores = preds.astype(np.float32)

        max_index = int(np.argmax(scores))

        # compute reward: need Delay & Jitter from inputs_bkp for chosen index
        chosen_norm = normalized_inputs[max_index].copy()
        chosen_norm['Delay'] = float(self.inputs_bkp[max_index]['Delay'])
        chosen_norm['Jitter'] = float(self.inputs_bkp[max_index]['Jitter'])
        reward = self.calculateReward(chosen_norm)

        # update per-protocol running stats with the chosen sample's Delay/Jitter
        proto = self.inputs_bkp[max_index].get('Protocol')
        if proto in self.protocol_stats:
            st = self.protocol_stats[proto]
            st_count = st['count'] + 1
            # running average update
            st['avg_delay'] = (st['avg_delay'] * st['count'] + float(self.inputs_bkp[max_index]['Delay'])) / st_count
            st['avg_jitter'] = (st['avg_jitter'] * st['count'] + float(self.inputs_bkp[max_index]['Jitter'])) / st_count
            st['count'] = st_count
            self.protocol_stats[proto] = st

        # store state_tuple (immutable), reward and episode id in memory
        chosen_state_tuple = tuple(prepared_inputs[max_index])
        self.memory.append((chosen_state_tuple, float(reward), int(self.episode_id)))

        # occasionally call train (deterministic or probabilistic)
        # here: train if warmup passed and at random 50% chance (you can change to always train)
        if len(self.memory) >= self.mem_warmup_steps and np.random.rand() > 0.5:
            self.modelTrain()

        return self.inputs[max_index]

    # ---------------------------
    # Model construction & prediction helper
    # ---------------------------
    def modelBuild(self, n_inputs=16, n_outputs=1):
        """
        Build a Keras model using Dense + BatchNorm + Dropout as suggested.
        Architecture: 64 -> 128 -> 64 -> 32 (with BatchNorm + Dropout)
        """
        inp = Input(shape=(n_inputs,))
        x = layers.Dense(64, activation=None)(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(128, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.20)(x)

        x = layers.Dense(64, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.15)(x)

        x = layers.Dense(32, activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        out = layers.Dense(n_outputs, activation='linear')(x)

        model = Model(inputs=inp, outputs=out)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.clipnorm)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def modelPrediction(self, model_inputs):
        """
        Return scalar float for a single state vector (length input_size).
        Kept for compatibility, but RMSE_RL uses batch predict for speed.
        """
        if np.random.rand() < self.epsilon:
            return float(np.random.rand())
        X = np.array([model_inputs], dtype=np.float32)
        pred = self.model.predict(X, verbose=0)
        return float(pred[0, 0])

    # ---------------------------
    # Training: n-step bootstrapped targets using target_model
    # ---------------------------
    def modelTrain(self):
        """
        Vectorized, stable training:
         - convert deque to list once
         - compute valid start indices (windows that do not cross episode boundaries)
         - sample starts by index (fast)
         - compute n-step return G and bootstrap with target_model on s_{t+n}
         - train main network on batch
         - periodic target_model updates
        """
        mem_len = len(self.memory)
        if mem_len < self.batch_size + self.window_size:
            return
        if mem_len < self.mem_warmup_steps:
            return

        mem_list = list(self.memory)  # snapshot

        # collect valid window start indices: start in [0, mem_len - window_size]
        valid_starts = []
        last_start = mem_len - self.window_size
        for s in range(0, last_start + 1):
            # ensure the whole window [s, s+window_size-1] is within same episode
            if mem_list[s][2] == mem_list[s + self.window_size - 1][2]:
                valid_starts.append(s)

        if len(valid_starts) == 0:
            return

        # sample start indices (without replacement)
        batch_k = min(self.batch_size, len(valid_starts))
        starts = random.sample(valid_starts, batch_k)

        X_batch = []
        y_batch = []

        for s in starts:
            # compute n-step return G = r0 + gamma*r1 + ... + gamma^{n-1} r_{n-1}
            G = 0.0
            for i in range(self.window_size):
                G += mem_list[s + i][1] * (self.gamma ** i)

            # bootstrap: V(s_{t+n}) from target_model if available and same episode
            next_index = s + self.window_size
            if next_index < mem_len and mem_list[next_index][2] == mem_list[s][2]:
                next_state = np.array(mem_list[next_index][0], dtype=np.float32).reshape(1, -1)
                v_next = float(self.target_model.predict(next_state, verbose=0)[0, 0])
                target_value = G + (self.gamma ** self.window_size) * v_next
            else:
                # terminal or out-of-range: pure n-step return
                target_value = G

            X_batch.append(np.array(mem_list[s][0], dtype=np.float32))
            y_batch.append(float(target_value))

        X_batch = np.vstack(X_batch)             # shape (batch_k, input_size)
        y_batch = np.array(y_batch).reshape(-1, 1)

        # train main network
        self.model.fit(X_batch, y_batch, epochs=1, verbose=0)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # update counters + target network
        self.train_updates += 1
        self.train_counter += 1

        # target network update: hard sync or optional soft update
        if self.soft_update_tau is None:
            if self.train_updates % self.target_update_freq == 0:
                self.target_model.set_weights(self.model.get_weights())
        else:
            # soft update (polyak averaging)
            tau = self.soft_update_tau
            main_w = self.model.get_weights()
            target_w = self.target_model.get_weights()
            new_w = [(tau * mw + (1.0 - tau) * tw) for mw, tw in zip(main_w, target_w)]
            self.target_model.set_weights(new_w)

    def saveModel(self, path="RMSE_RL_improved.keras"):
        self.model.save(path)
        # optionally save target model or protocol_stats separately if desired
