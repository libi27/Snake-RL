import numpy as np
import random

from policies import base_policy as bp

EPSILON = 0.8
EPSILON_RATE = 0.95
EPSILON_MIN = 0.05
GAMMA = 0.6
ETA = 0.1

DIRECTIONS = ['E', 'W', 'N', 'S']
MOVES = [[], ['L'], ['R'], ['F'], ['F', 'L'], ['F', 'R'], ['L', 'L'], ['R', 'R']]
NUM_OF_FEATURES = 11


class Linear200024677(bp.Policy):
    """
    A simple policy not storing any data, just performing a linear learner updated each learning round
    """

    def cast_string_args(self, policy_args):
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['eta'] = float(policy_args['eta']) if 'eta' in policy_args else ETA
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['epsilon_rate'] = float(
            policy_args['epsilon_rate']) if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['epsilon_min'] = float(policy_args['epsilon_min']) if 'epsilon_min' in policy_args else EPSILON_MIN
        return policy_args

    def init_run(self):
        '''
        init our linear model and the moving steps in each direction

        :return: no returned values
        '''
        self.len_range_on_borad = NUM_OF_FEATURES
        self.num_of_features = 1 + self.len_range_on_borad * len(MOVES)
        self.model = np.random.randn(self.num_of_features)
        self.board_width, self.board_height = self.board_size
        self.moves_calc = dict()
        self.moves_calc['N'] = self.get_moves_for_dir('N')
        self.moves_calc['W'] = self.get_moves_for_dir('W')
        self.moves_calc['S'] = (-1) * self.moves_calc['N']
        self.moves_calc['E'] = (-1) * self.moves_calc['W']

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
        if prev_state is None:
            return np.random.choice(bp.Policy.ACTIONS)
        if random.random() < self.epsilon:
            return self.ACTIONS[random.randint(0, len(self.ACTIONS) - 1)]
        else:
            action, value = self.predict(new_state)
            return action

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        '''
        perfrom our implementation of a linear q-learning function

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
        if prev_state is None:
            return

        _, q_new_s_a = self.predict(new_state)
        q_target = reward + (self.gamma * q_new_s_a)
        q_s_a, s_rep = self.Q_s_a(prev_state, prev_action)
        delta = q_target - q_s_a
        self.model += self.eta * delta * s_rep
        self.epsilon = self.epsilon * self.epsilon_rate if self.epsilon > self.epsilon_min else self.epsilon_min

    @staticmethod
    def get_moves_for_dir(direction):
        '''
        calculate the moves for a given direction
        :param direction:
        :return:
        '''
        arr = np.empty((len(MOVES), 2), dtype=int)  # all moves including current pos
        ind = 0
        for move in MOVES:
            move_dir = direction
            pos = PositionCounter([0, 0])
            if len(move) == 0:
                c = 0
                r = 0
            else:
                for int_act in move:
                    move_dir = bp.Policy.TURNS[move_dir][int_act]
                    pos = pos.move(move_dir)
                    r = pos[0]
                    c = pos[1]
            arr[ind, :] = [r, c]
            ind += 1
        return arr

    def get_state_representation(self, state, action):
        '''
        calculates a one hot vector for each position around the head
        :param state: current board state
        :param action: the action to perform
        :return: feature vector representing the state of the nearest neighboring of the snake head after performing the action
        '''
        board, head = state
        head_pos, head_direction = head
        representation = np.zeros(self.num_of_features)
        representation[0] = 1  # bias
        index = 1

        new_dir = bp.Policy.TURNS[head_direction][action]
        next_position = head_pos.move(new_dir)

        r_next = next_position[0]
        c_next = next_position[1]

        moves = np.add(np.array([r_next, c_next]), self.moves_calc[new_dir])
        moves_rows = np.remainder(moves[:, 0], self.board_width)
        moves_cols = np.remainder(moves[:, 1], self.board_height)
        indices = (moves_rows, moves_cols)
        moves_on_board = board[indices]
        moves_on_board += 1
        one_hot = np.eye(self.len_range_on_borad)[moves_on_board]
        all_one = np.reshape(one_hot, (8 * self.len_range_on_borad))

        representation[index:index + len(all_one)] = all_one

        return representation

    def Q_s_a(self, state, action):
        '''
        Q-function predicting result of a state and given action
        :param state:
        :param action:
        :return: prediction and state representation after action performed
        '''
        s_a_rep = self.get_state_representation(state, action)
        return np.dot(s_a_rep, self.model), s_a_rep

    def predict(self, state):
        '''
        predict the action given the state
        :param state:
        :return: action and predicted value
        '''
        all_predictions = np.zeros(len(self.ACTIONS))
        for indx_action, action in enumerate(self.ACTIONS):
            all_predictions[indx_action], _ = self.Q_s_a(state, action)
        index_action = np.argmax(all_predictions)
        return self.ACTIONS[index_action], all_predictions[index_action]


class PositionCounter:
    '''
    follow up the indexes in feature vector
    '''
    def __init__(self, position):
        self.pos = position

    def __getitem__(self, key):
        return self.pos[key]

    def __add__(self, other):
        return PositionCounter(((self[0] + other[0]),
                                (self[1] + other[1])))

    def move(self, dir):
        '''
        return indices after moving in direction dir
        :param dir:
        :return:
        '''
        if dir == 'E':
            return self + (0, 1)
        if dir == 'W':
            return self + (0, -1)
        if dir == 'N':
            return self + (-1, 0)
        if dir == 'S':
            return self + (1, 0)
        raise ValueError('unrecognized direction')
