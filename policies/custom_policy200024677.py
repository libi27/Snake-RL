import numpy as np
import random
import tensorflow as tf

from policies import base_policy as bp

# learning flags
EPSILON = 0.8
EPSILON_RATE = 0.95
EPSILON_MIN = 0.05
SIZE_OF_BATCH = 64
MIN_BATCH_SIZE = 32
MAX_BATCH_SIZE = 256
MAX_SIZE_DB = 1024

GAMMA = 0.8
FEATURE_SIZE = 100
MIN_LEARNING_TIMES = 4
ACTIONS_INDX = {'L': 0,  # counter clockwise (left)
                'R': 1,  # clockwise (right)
                'F': 2}  # forward

INSIDE_MASK_SIZE = 6

DIRECTIONS = ['E', 'W', 'N', 'S']


class Custom200024677(bp.Policy):
    """
    Custom policy implementating our approach to q-learning function with our hard coded tuned parameters, model definition
    and state representation
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['epsilon_rate'] = float(
            policy_args['epsilon_rate']) if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['epsilon_min'] = float(policy_args['epsilon_min']) if 'epsilon_min' in policy_args else EPSILON_MIN
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else SIZE_OF_BATCH
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['inside_mask'] = int(
            policy_args['inside_mask']) if 'inside_mask' in policy_args else INSIDE_MASK_SIZE
        return policy_args

    def init_run(self):
        '''
        init the masks we need in order to represent the tuple (state, board) in our representation.
        Init the NN we use as a q-function approxiamation
        :return:
        '''
        [self.board_width, self.board_height] = self.board_size
        self.range_on_board = np.arange(-1, 10)
        self.len_range_on_board = len(self.range_on_board)

        mask1 = np.array([[False, False, False, False, False],
                          [False, True, True, True, False, ],
                          [False, True, True, True, False, ],
                          [False, True, False, True, False],
                          [False, False, False, False, False]])
        mask2 = np.rot90(mask1)
        self.board_masks = dict()
        self.board_masks['N'] = mask1
        self.board_masks['W'] = mask2
        self.board_masks['S'] = np.flip(mask1, axis=0)
        self.board_masks['E'] = np.flip(mask2, axis=0)

        mask_s1 = np.array([[True, True, True, True, True],
                            [True, False, False, False, True],
                            [True, False, False, False, True],
                            [True, False, False, False, True],
                            [True, True, True, True, True]])
        mask_s2 = np.rot90(mask_s1)
        self.board_masks_s = dict()
        self.board_masks_s['N'] = mask_s1
        self.board_masks_s['W'] = mask_s2
        self.board_masks_s['S'] = np.flip(mask_s1, axis=0)
        self.board_masks_s['E'] = np.flip(mask_s2, axis=0)

        if self.inside_mask * 2 + 1 >= self.board_height or self.inside_mask * 2 + 1 >= self.board_width:
            self.inside_mask = int(min(self.board_height, self.board_width) / 2 - 1)
        self.f_mask = Mask_feature((self.inside_mask, self.inside_mask), self.len_range_on_board)

        self.num_of_features = self.len_range_on_board * len(mask1[mask1]) + self.len_range_on_board \
                               + self.f_mask.num_of_features() + len(DIRECTIONS)

        self.model = create_dqn_model(self.num_of_features)
        self.db = ExperienceReplay(self.num_of_features, len(self.ACTIONS))
        self.train_mode = False

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        '''
        perfrom our implementation of q-learning

        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example)
        '''
        if self.train_mode:
            if too_slow:
                self.batch_size = max(self.batch_size//2, MIN_BATCH_SIZE)
            else:
                self.batch_size = min(self.batch_size+2, MAX_BATCH_SIZE)
            batch = self.db.sample_batch(self.batch_size)
            new_state_value_predict = self.model.predict_on_batch(
                np.reshape(batch['new_state'], (self.batch_size * 3, self.num_of_features)))

            new_state_value_predict = np.squeeze(new_state_value_predict)
            new_state_value_predict_max = np.reshape(new_state_value_predict, (self.batch_size, len(self.ACTIONS))).max(
                axis=1)

            q_target = batch['reward'] + (self.gamma * np.reshape(new_state_value_predict_max, (self.batch_size, 1)))
            y = np.reshape(q_target, (self.batch_size, 1))
            self.model.train_on_batch(batch['prev_state'], y)
        else:
            self.train_mode = self.db.size() >= self.batch_size

        self.epsilon = self.epsilon * self.epsilon_rate if self.epsilon > self.epsilon_min else self.epsilon_min

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        '''
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        '''
        if too_slow:
            self.batch_size = max(self.batch_size//2, MIN_BATCH_SIZE)
        else:
            self.batch_size = min(self.batch_size+2, MAX_BATCH_SIZE)
        last_rep = self.get_next_state_representation(new_state)
        if prev_state is not None:
            self.db.save_state(self.get_representation(prev_state, prev_action), reward, last_rep)
        if random.random() < self.epsilon:
            return self.ACTIONS[random.randint(0, len(self.ACTIONS) - 1)]
        else:
            return self.ACTIONS[self.calc_qa_max_action(last_rep)]

    def calc_qa_max_action(self, state):
        '''
        predict on batch of triplet of state which are the state possiblities then max on the three results
        :param state:
        :return:
        '''
        result = self.model.predict(state, batch_size=3)
        return np.argmax(result)

    def get_small_area(self, state, next_position, next_head_dir):
        '''
        crop the board around the snake head
        :param state:
        :param next_position:
        :param next_head_dir:
        :return:
        '''
        crop = ExperienceReplay.crop_board(2, 2, state, next_position)
        mask = self.board_masks[next_head_dir]

        values = crop[mask, ...] + 1
        one_hot = np.eye(self.len_range_on_board)[values]
        all_one = np.reshape(one_hot, (8 * self.len_range_on_board))

        mask2 = self.board_masks_s[next_head_dir]

        values2 = crop[mask2, ...] + 1
        all2 = np.zeros(self.len_range_on_board)
        all2[values2] = 1

        return np.hstack((all_one, all2))

    def get_representation(self, state, action):
        '''
        calculates a one hot vector for each position around the head
        :param state: current board state
        :param action: the action to perform
        :return: feature vector representing the state of the nearest neighboring of the snake head after performing the action
        '''
        board, head = state
        head_pos, head_direction = head

        representation = np.zeros(self.num_of_features)
        indx = 0
        new_dir = bp.Policy.TURNS[head_direction][action]
        next_position = head_pos.move(new_dir)
        features = self.get_small_area(state, next_position, new_dir)
        representation[indx:indx + len(features)] = features
        indx = indx + len(features)

        small_board = ExperienceReplay.crop_board(self.f_mask.size_of_mask_x, self.f_mask.size_of_mask_y, state,
                                                  next_position)
        indx = self.f_mask.update_representation(small_board, new_dir, representation, indx)

        representation[indx] = int(head_direction == DIRECTIONS[0])
        representation[indx + 1] = int(head_direction == DIRECTIONS[1])
        representation[indx + 2] = int(head_direction == DIRECTIONS[2])
        representation[indx + 3] = int(head_direction == DIRECTIONS[3])
        indx += 4

        return representation

    def get_next_state_representation(self, next_state):
        '''
        compute the state for each directions
        :param next_state:
        :return:
        '''
        all_next_state_actions = np.empty((len(self.ACTIONS), self.num_of_features))
        for ind, action in enumerate(self.ACTIONS):
            all_next_state_actions[ind] = self.get_representation(next_state, action)
        return all_next_state_actions


class ExperienceReplay:
    '''
    class to handle our stored state and there representation as well as the sampling process
    '''

    def __init__(self, feature_size, action_size, max_exp_size=MAX_SIZE_DB):
        '''
        init the containers for our feature vectors.
        :param feature_size:
        :param action_size:
        :param max_exp_size:
        '''
        self._max_exp_size = max_exp_size
        self._new_state = np.zeros((self._max_exp_size, action_size, feature_size))
        self._prev_state = np.zeros((self._max_exp_size, feature_size))
        self._action = np.zeros((self._max_exp_size, action_size), dtype=bool)
        self._reward = np.zeros((self._max_exp_size, 1))
        self._save_counter = 0
        self._insert_idx = 0

    def save_state(self, prev_state, reward, new_state):
        '''
        save in modulo way the states
        :param prev_state:
        :param reward:
        :param new_state:
        :return:
        '''
        self._new_state[self._insert_idx, :] = new_state
        self._prev_state[self._insert_idx, :] = prev_state
        self._reward[self._insert_idx, :] = reward
        self._save_counter += 1
        self._insert_idx = (self._insert_idx + 1) % self._max_exp_size

    def sample_batch(self, batch_size=50):
        '''
        perform a sampling process without replacement
        :param batch_size:
        :return:
        '''
        indices_of_batch = np.random.choice(self.size(), batch_size, replace=False)
        return {'prev_state': self._prev_state[indices_of_batch, :],
                'action': self._action[indices_of_batch, :],
                'reward': self._reward[indices_of_batch, :],
                'new_state': self._new_state[indices_of_batch, :]}

    def get_last_state(self):
        '''
        return the last instance saved
        :return:
        '''
        return self._new_state[self._insert_idx - 1, :]

    def size(self):
        '''
        number of saved state until now
        :return:
        '''
        return min(self._save_counter, self._max_exp_size)

    @staticmethod
    def crop_board(inside_mask_x, inside_mask_y, new_state, next_position):
        '''
        crop the board
        :param inside_mask_x:
        :param inside_mask_y:
        :param new_state:
        :param next_position:
        :return:
        '''
        board, _ = new_state
        r_next, c_next = next_position.pos

        tmp_board = np.array(board)

        if (r_next - inside_mask_x) < 0:
            tmp_board = np.roll(tmp_board, inside_mask_x, axis=0)
            r_next += inside_mask_x

        if (r_next + inside_mask_x) >= board.shape[0] - 1:
            tmp_board = np.roll(tmp_board, -((r_next + inside_mask_x) % board.shape[0]), axis=0)
            r_next -= ((r_next + inside_mask_x) % (board.shape[0] - 1))

        if (c_next - inside_mask_y) < 0:
            tmp_board = np.roll(tmp_board, inside_mask_y, axis=1)
            c_next += inside_mask_y

        if (c_next + inside_mask_y) >= board.shape[1] - 1:
            tmp_board = np.roll(tmp_board, -((c_next + inside_mask_y) % (board.shape[0] - 1)), axis=1)
            c_next -= ((c_next + inside_mask_y) % (board.shape[1] - 1))

        r1 = r_next - inside_mask_x
        r2 = r_next + inside_mask_x
        c1 = c_next - inside_mask_y
        c2 = c_next + inside_mask_y
        crop = tmp_board[r1:r2 + 1, c1:c2 + 1]
        return crop


def create_dqn_model(feature_size):
    '''
    generate the nn model
    :param feature_size:
    :return:
    '''
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(feature_size,)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # model.summary()
    return model


class Mask_feature:
    '''
    class to hold the mask we used on the cropped board to retrieve our feature vector optimaly
    '''
    def __init__(self, size_of_mask, len_range_on_borad):
        '''

        :param size_of_mask:
        :param len_range_on_borad:
        '''
        self.size_of_mask_x, self.size_of_mask_y = size_of_mask
        self.len_range_on_board = len_range_on_borad
        # build masks
        center_x = self.size_of_mask_x
        center_y = self.size_of_mask_y
        mask1 = np.zeros((center_x * 2 + 1, center_y * 2 + 1), dtype=bool)
        mask1[:center_x + 1, center_y:] = True
        mask2 = np.zeros((center_x * 2 + 1, center_y * 2 + 1), dtype=bool)
        mask2[:center_x + 1, :center_y] = True
        mask3 = np.zeros((center_x * 2 + 1, center_y * 2 + 1), dtype=bool)
        mask3[center_x + 1:, :center_y] = True
        mask4 = np.zeros((center_x * 2 + 1, center_y * 2 + 1), dtype=bool)
        mask4[center_x + 1:, center_y:] = True

        self.board_masks = {}
        self.board_masks['N'] = np.array([mask1, mask2, mask3, mask4], dtype=bool)
        self.board_masks['W'] = np.array([mask2, mask3, mask4, mask1], dtype=bool)
        self.board_masks['S'] = np.array([mask3, mask4, mask1, mask2], dtype=bool)
        self.board_masks['E'] = np.array([mask4, mask1, mask2, mask3], dtype=bool)

    def num_of_features(self):
        '''
        return the num of features
        :return:
        '''
        return len(self.board_masks) * self.len_range_on_board

    def update_representation(self, small_board, new_dir, representation_vector, index_to_start):
        '''
        
        :param small_board:
        :param new_dir:
        :param representation_vector:
        :param index_to_start:
        :return:
        '''
        indx = index_to_start
        for mask in self.board_masks[new_dir]:
            values = small_board[mask, ...]
            results, counts = np.unique(values, return_counts=True)
            results += 1
            existing_in_mask = np.zeros(self.len_range_on_board)
            existing_in_mask[results] = counts > 0
            representation_vector[indx:indx + self.len_range_on_board] = existing_in_mask
            indx += self.len_range_on_board
        return indx
