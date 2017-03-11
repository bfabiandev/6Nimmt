import random
import numpy as np
import AI
import plotly.plotly as py
import matplotlib.pyplot as plt

def get_size_of_penalty(penalty_cards):
    penalty = 0
    for card in penalty_cards:
        if card == 55:
            penalty += 7
        elif card % 11 == 0:
            penalty += 5
        elif card % 10 == 0:
            penalty += 3
        elif card % 5 == 0:
            penalty += 2
        else:
            penalty += 1
    return penalty


class Deck:
    cards = None

    def __init__(self):
        self.cards = list(range(1, 105))
        np.random.shuffle(self.cards)

    def get_random_card(self):
        return self.cards.pop()

    def __str__(self):
        return str(self.cards)


class Player:
    cards = None
    penalty = None
    random = None

    def __init__(self, random=False):
        self.cards = []
        self.penalty = 0
        self.random = random

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)

    def choose_card(self, game):
        if not self.random:
            state = game.get_state()
            qvals = model.predict(state.reshape(1, 456), batch_size=1)
            indexes = self.cards.copy()
            indexes[:] = [x - 1 for x in indexes]
            possible_qvals = qvals[:, indexes]
            max_index = np.argmax(possible_qvals)
            chosen = self.cards[max_index]
        else:
            chosen = np.random.choice(self.cards)

        self.remove_card(chosen)
        return chosen

    def add_to_penalty(self, penaltyCards):
        self.penalty += get_size_of_penalty(penaltyCards)

    def __str__(self):
        return str(self.cards)


class Board:
    slots = None

    def __init__(self):
        self.slots = []

    def fill_slots(self, deck):
        for i in range(4):
            self.slots.append([[deck.get_random_card()], 1])
        self.slots.sort(key=lambda tup: tup[0][-1])

    def choose_slot(self):
        min_penalty = 1000
        best_slot_index = -1
        for i, slot in enumerate(self.slots):
            penalty = get_size_of_penalty(slot[0])
            if penalty < min_penalty:
                min_penalty = penalty
                best_slot_index = i
        return best_slot_index

    def __str__(self):
        return str(self.slots)

    def add_card(self, card):
        index = 3
        penalty_cards = []

        for i, slot in enumerate(self.slots):
            if slot[0][-1] > card[0]:
                index = i - 1
                break

        if index == -1:
            # index = np.random.randint(0, 4)
            index = self.choose_slot()
            penalty_cards = self.slots[index][0]
            self.slots[index] = [[card[0]], 1]
            self.slots.sort(key=lambda tup: tup[0][-1])

        else:
            if self.slots[index][1] == 5:
                penalty_cards = self.slots[index][0]
                self.slots[index] = [[card[0]], 1]
                self.slots.sort(key=lambda tup: tup[0][-1])
            else:
                self.slots[index][0].append(card[0])
                self.slots[index][1] += 1

        return penalty_cards


class Game:
    deck = None
    players = None
    board = None

    def __init__(self, number_of_players=2, random_players=0):
        self.deck = Deck()
        self.players = []
        self.board = Board()
        self.deal_hands(number_of_players, random_players)
        self.board.fill_slots(deck=self.deck)

    def deal_hands(self, number_of_players=2, random_players=0):
        if number_of_players > 10 or number_of_players < 1:
            print("Not a valid number of players. [2-10]")

        for i in range(number_of_players-random_players):
            self.players.append(Player(random=False))
        for i in range(random_players):
            self.players.append(Player(random=True))

        for hand in self.players[:]:
            for i in range(10):
                hand.add_card(self.deck.get_random_card())
            hand.cards.sort()

    def print_hands(self):
        for i, hand in enumerate(self.players): print("Player {}'s hand: {}".format(i, hand))

    def print_deck(self):
        print("Deck: " + str(self.deck))

    def print_board(self):
        print("Board: " + str(self.board))

    def play_game(self, history=None, verbose=False):
        if history is None:
            history = []

        for i in range(10):
            self.play_round(history, verbose)
            if verbose:
                self.print_hands()
                self.print_board()

    def play_round(self, history=None, verbose=False):
        if history is None:
            history = []

        old_state = self.get_state()
        old_cards = []
        for i in self.players:
            old_cards.append(i.cards.copy())

        cards_this_turn = []
        for i, hand in enumerate(self.players):
            chosen_card = hand.choose_card(self)
            cards_this_turn.append((chosen_card, i))
            if verbose:
                print("{}. player pick card {}.".format(i, chosen_card))

        penalties = self.put_cards_on_board(cards_this_turn, verbose)
        for penalty in penalties:
            for card in cards_this_turn:
                if penalty[1] == card[1]:
                    new_cards = self.players[card[1]].cards.copy()
                    history.append((old_state, old_cards[card[1]], card[0], -penalty[0], self.get_state(), new_cards))

    def put_cards_on_board(self, cards, verbose=False):
        penalties = []
        cards.sort(key=lambda tup: tup[0])
        for card in cards:
            penalty_cards = self.board.add_card(card)
            self.players[card[1]].add_to_penalty(penalty_cards)
            penalties.append((get_size_of_penalty(penalty_cards), card[1]))

            if len(penalty_cards) > 0 and verbose:
                print("{}. Player's penalty increased by {}.".format(card[1], get_size_of_penalty(penalty_cards)))

        return penalties

    def get_state(self):
        state = np.empty(shape=(0, 0))
        for slot in self.board.slots:
            state = np.append(state, state_to_one_hot(slot[0][-1], 104))
            state = np.append(state, state_to_one_hot(slot[1], 5))

        state = np.append(state, state_to_one_hot(len(self.players), 10))
        state = np.append(state, state_to_one_hot(len(self.players[0].cards), 10))
        return state

    def print_penalties(self):
        for i, player in enumerate(self.players):
            print("{}. Player's penalty: {}".format(i, player.penalty))

    def get_penalties(self):
        penalties = np.zeros(len(self.players))
        for i, player in enumerate(self.players):
            penalties[i] = player.penalty
        return penalties

def state_to_one_hot(number, num_categories):
    arr = np.zeros(shape=(num_categories))
    arr[number - 1] = 1
    return arr


def train():
    epochs = 2000
    gamma = 0.25
    history_size = 5000
    history = []

    for i in range(epochs):
        batch_size = 500
        game = Game(np.random.randint(2, 11))
        game.play_game(history)
        if len(history) > history_size:
            history = history[-history_size:]
        if batch_size > len(history):
            batch_size = len(history)
        minibatch = random.sample(history, batch_size)
        x_train = []
        y_train = []
        for memory in minibatch:
            old_state, old_cards, action, reward, new_state, new_cards = memory
            old_qvals = model.predict(old_state.reshape(1, 456), batch_size=1)

            y = np.zeros((1, 104))
            y[:] = old_qvals[:]

            if len(new_cards) == 0:
                update = reward
            else:
                new_qvals = model.predict(new_state.reshape(1, 456), batch_size=1)

                indexes = new_cards.copy()
                indexes[:] = [x - 1 for x in indexes]
                possible_qvals = new_qvals[:, indexes]
                max_new_qval = np.max(possible_qvals)
                update = reward + gamma * max_new_qval

            y[0][action - 1] = update
            x_train.append(old_state.reshape(456, ))
            y_train.append(y.reshape(104, ))

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        print("Game #: %s" % (i,))
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=1, verbose=1)


model = AI.load_model()

#train()

AI.save_model(model)

number_of_games = 10000
statistics = np.empty(shape=(number_of_games,10), dtype=int)
for i in range(number_of_games):
    print("Game #: {}".format(i))
    game = Game(10, random_players=8)
    game.play_game(verbose=False)
    pens = game.get_penalties()
    statistics[i, :] = pens

mean = np.mean(statistics, axis=0)
variance = np.var(statistics, axis=0)

for i in range(len(mean)):
    print("Player {}'s scores have a mean of {} and variance of {}.".format(i, mean[i], variance[i]))
#fig = plt.figure()

#n, bins, patches = plt.hist(mean)

#plt.show()