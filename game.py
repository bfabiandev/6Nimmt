import numpy as np


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

    def chooseCard(self, board):
        chosen = np.random.choice(self.cards)
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
                index = i-1
                break

        if index == -1:
            index = np.random.randint(0,4)
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

    def playGame(self):
        for i in range(10):
            self.playRound()
            game.printHands()
            game.printBoard()

    def playRound(self):
        cardsThisTurn = []
        for i, hand in enumerate(self.players):
            chosenCard = hand.chooseCard(self.board)
            cardsThisTurn.append((chosenCard, i))
            print("{}. player pick card {}.".format(i, chosenCard))
        self.putCardsOnBoard(cardsThisTurn)


    def putCardsOnBoard(self, cards):
        cards.sort(key=lambda tup: tup[0])
        for card in cards:
            penaltyCards = self.board.addCard(card)
            self.players[card[1]].addToPenalty(penaltyCards)
            if len(penaltyCards) > 0: print("{}. Player's penalty increased by {}.".format(card[1], getSizeOfPenalty(penaltyCards)))


    def printPenalties(self):
        for i, player in enumerate(self.players):
            print("{}. Player's penalty: {}".format(i, player.penalty))


game = Game(10)
game.printHands()
game.printBoard()
game.playGame()
game.printPenalties()