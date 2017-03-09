import random
import numpy as np
import AI

def getSizeOfPenalty(penaltyCards):
    penalty = 0
    for card in penaltyCards:
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

    def getRandomCard(self):
        return self.cards.pop()

    def __str__(self):
        return str(self.cards)


class Player:
    cards = None
    penalty = None

    def __init__(self):
        self.cards = []
        self.penalty = 0

    def addCard(self, card):
        self.cards.append(card)

    def removeCard(self, card):
        if card in self.cards:
            self.cards.remove(card)

    def chooseCard(self, game, board):
        state = np.empty(shape=(0, 0))
        for slot in board.slots:
            state = np.append(state, [slot[0][-1], slot[1]])

        state = np.append(state, [len(game.players)])
        state = np.append(state, [len(self.cards)])
        qvals = model.predict(state.reshape(1, 10), batch_size=1)
        indexes = self.cards.copy()
        indexes[:] = [x - 1 for x in indexes]
        possibleQvals = qvals[:, indexes]
        maxIndex = np.argmax(possibleQvals)
        chosen = self.cards[maxIndex]

        self.removeCard(chosen)
        return chosen

    def addToPenalty(self, penaltyCards):
        self.penalty += getSizeOfPenalty(penaltyCards)

    def __str__(self):
        return str(self.cards)


class Board:
    slots = None

    def __init__(self):
        self.slots = []

    def fillSlots(self, deck):
        for i in range(4):
            self.slots.append([[deck.getRandomCard()], 1])
        self.slots.sort(key=lambda tup: tup[0][-1])

    def __str__(self):
        return str(self.slots)

    def addCard(self, card):
        index = 3
        penaltyCards = []

        for i, slot in enumerate(self.slots):
            if slot[0][-1] > card[0]:
                index = i - 1
                break

        if index == -1:
            index = np.random.randint(0, 4)
            penaltyCards = self.slots[index][0]
            self.slots[index] = [[card[0]], 1]
            self.slots.sort(key=lambda tup: tup[0][-1])

        else:
            if self.slots[index][1] == 5:
                penaltyCards = self.slots[index][0]
                self.slots[index] = [[card[0]], 1]
                self.slots.sort(key=lambda tup: tup[0][-1])
            else:
                self.slots[index][0].append(card[0])
                self.slots[index][1] += 1

        return penaltyCards


class Game:
    deck = None
    players = None
    board = None

    def __init__(self, numberOfPlayers=2):
        self.deck = Deck()
        self.players = []
        self.board = Board()
        self.dealHands(numberOfPlayers)
        self.board.fillSlots(deck=self.deck)

    def dealHands(self, numberOfPlayers=2):
        if numberOfPlayers > 10 or numberOfPlayers < 1:
            print("Not a valid number of players. [2-10]")

        for i in range(numberOfPlayers):
            self.players.append(Player())

        for hand in self.players[:]:
            for i in range(10):
                hand.addCard(self.deck.getRandomCard())
            hand.cards.sort()

    def printHands(self):
        for i, hand in enumerate(self.players): print("Player {}'s hand: {}".format(i, hand))

    def printDeck(self):
        print("Deck: " + str(self.deck))

    def printBoard(self):
        print("Board: " + str(self.board))

    def playGame(self, history=None):
        if history is None:
            history = []

        for i in range(10):
            self.playRound(history)
            self.printHands()
            self.printBoard()

    def playRound(self, history=None):
        if history is None:
            history = []

        oldState = self.getState()
        oldCards = []
        for i in self.players:
            oldCards.append(i.cards.copy())

        cardsThisTurn = []
        for i, hand in enumerate(self.players):
            chosenCard = hand.chooseCard(self, self.board)
            cardsThisTurn.append((chosenCard, i))
            print("{}. player pick card {}.".format(i, chosenCard))

        penalties = self.putCardsOnBoard(cardsThisTurn)
        for penalty in penalties:
            for card in cardsThisTurn:
                if penalty[1] == card[1]:
                    newCards = self.players[card[1]].cards.copy()
                    history.append((oldState, oldCards[card[1]], card[0], -penalty[0], self.getState(), newCards))

    def putCardsOnBoard(self, cards):
        penalties = []
        cards.sort(key=lambda tup: tup[0])
        for card in cards:
            penaltyCards = self.board.addCard(card)
            self.players[card[1]].addToPenalty(penaltyCards)
            penalties.append((getSizeOfPenalty(penaltyCards), card[1]))

            if len(penaltyCards) > 0:
                print("{}. Player's penalty increased by {}.".format(card[1], getSizeOfPenalty(penaltyCards)))

        return penalties

    def getState(self):
        state = np.empty(shape=(0, 0))
        for slot in self.board.slots:
            state = np.append(state, [slot[0][-1], slot[1]])

        state = np.append(state, [len(self.players)])
        state = np.append(state, [len(self.players[0].cards)])
        return state

    def printPenalties(self):
        for i, player in enumerate(self.players):
            print("{}. Player's penalty: {}".format(i, player.penalty))


def train():
    epochs = 5000
    gamma = 0.25
    historySize = 500
    history = []

    for i in range(epochs):
        batchSize = 100
        game = Game(np.random.randint(2, 11))
        game.playGame(history)
        if len(history) > historySize:
            history = history[-historySize:]
        if batchSize > len(history):
            batchSize = len(history)
        minibatch = random.sample(history, batchSize)
        X_train = []
        Y_train = []
        for memory in minibatch:
            oldState, oldCards, action, reward, newState, newCards = memory
            oldQvals = model.predict(oldState.reshape(1,10), batch_size=1)

            y = np.zeros((1,104))
            y[:] = oldQvals[:]

            if len(newCards)==0:
                update = reward
            else:
                newQvals = model.predict(newState.reshape(1, 10), batch_size=1)

                indexes = newCards.copy()
                indexes[:] = [x - 1 for x in indexes]
                possibleQvals = newQvals[:, indexes]
                maxNewQval = np.max(possibleQvals)
                update = reward + gamma*maxNewQval

            y[0][action-1] = update
            X_train.append(oldState.reshape(10,))
            Y_train.append(y.reshape(104,))


        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        print("Game #: %s" % (i,))
        model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=1, verbose=1)


model = AI.loadModel()

#train()

AI.saveModel(model)

game = Game(4)
game.printHands()
game.printBoard()
game.playGame()
game.printPenalties()