import os
import torch

from .train import Agent, state_to_features, ACTIONS, warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# das ist eine Memberfunction des Agent, die muss in der Klasse beleiben
# def load_model(self):


def setup(self):
    continueTraining = False
    model_path = './'

    self.agent = Agent(gamma=0.99, epsilon= 1, lr=1e-5,
                          input_dims=(17, 17, 1), eps_dec=1e-4,
                          n_actions=6, max_mem_size=10000, batch_size=64,
                          eps_end=0.01, replace=1000, fname=model_path, train=self.train)

    device = torch.device('cuda' if torch.cuda.is_available() and self.train else 'cpu')
    if (not continueTraining and self.train):
    # TODO: fix file path. This does not work
        self.logger.info("Setting up model from scratch.")
        print("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        print("Loading model from saved state.")
        self.agent.load_model(mode=device)


def act(self, game_state: dict) -> str:
    # if np.random.random() < self.agent.epsilon:
    # 	return np.random.choice(ACTIONS)
    # # maybe?
    # # bitte so umschreiben, dass das nur nen game state braucht
    # actions = self.q_net(state_to_features(game_state)[np.newaxis, :])
    # action = np.argmax(actions)
    # return str(action)
    observation = state_to_features(game_state)
    if self.train and game_state['round'] < self.warmup:# and not os.path.exists('observation.pt'):
        self.current_round = game_state['round']
        a = warmup(self, game_state)
        #print("warming up", self.current_round,a if a != None else 'BOMB')
        return a if a != None else 'WAIT'
    else:
        a = ACTIONS[self.agent.choose_action(observation)]
        #a = 'BOMB'
        return a
