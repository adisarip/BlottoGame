# Authors: Lokesh Balani, Aditya Saripalli, 2018
#
# A simulator for a Weighted Colonel Blotto game that is played as follows:
#
# We are given battlefields labelled 1-n, and we assume that each battlefield has the same value.
# Two players present attack strategies given a fixed number of soldiers, deciding how many
# soldiers they want to send to every battlefield. Once the bets are compared, the player with
# the most number soldiers at a battlefield wins that battlefield's points.
# For 4 battlefield and 10 soldiers, this might look like:
#
#  Player 1's bets (Attacker) :   4   2   3   1
#  --------------------------------------------
#  Player 2's bets (Defender) :   0   0   2   8
#
#
# Player 1 is assigned a score of 3, because the player won 3 battlefields.
# Player 2 scores 1 for winning 1 Battlefield
# Hence Player-2 is the winner.
#

import random
import csv
import operator as op
from collections import defaultdict
from functools import reduce


def create_strategy(n_soldiers,
                    n_battlefields):
    # Create a random strategy
    res = n_soldiers
    # Empty list for list of strategies
    strategy = []
    for i in range(n_battlefields - 1):
        # Fill each position in strategy with a random number
        # between 0 and remaining reserves
        rand = random.randint(0, res)
        # add this to the strategy
        strategy.append(rand)
        # Reduce reserves by the allocated number
        res -= rand
    # Add any remaining reserves at the last
    strategy.append(res)
    # Shuffle the strategies and return
    random.shuffle(strategy)

    return strategy


def validate_strategy(strategy,
                      d_unique_strategies):
    is_valid = False
    # verify the strategy string
    value_str = ""
    for value in strategy:
        value_str += str(value) + "_"
    value_str = value_str[:-1]
    value_str_len = len(value_str)

    # check for the uniqueness of the strategy string and append if unique
    if value_str_len in d_unique_strategies.keys():
        if value_str not in d_unique_strategies[value_str_len]:
            d_unique_strategies[value_str_len].append(value_str)
            is_valid = True
    else:
        d_unique_strategies[value_str_len] = [value_str]
        is_valid = True

    return is_valid


# Create a unique integer-valued strategy that sums to
# No of soldiers with length equal to number of battlefields
def create_unique_strategy(n_soldiers,
                           n_battlefields,
                           d_unique_strategies):
    is_valid = False
    strategy = []
    while not is_valid:
        strategy = create_strategy(n_soldiers, n_battlefields)
        # Validate the strategies
        assert sum(strategy) == n_soldiers
        is_valid = validate_strategy(strategy,
                                     d_unique_strategies)
    return strategy


class Blotto:
    def __init__(self,
                 n_sol,
                 n_bfs):
        """ Set the number of soldiers available and the number of battlefields """
        self.n_soldiers = int(n_sol)
        self.n_battlefields = int(n_bfs)
        self.master_dataset = []
        self.unique_strategies = defaultdict(list)
        self.strategy_space_size = min(self.compute_strategy_space_size(), 10000)
        print("#Soldiers : {}".format(self.n_soldiers))
        print("#Battlefields : {}".format(self.n_battlefields))

    def compute_strategy_space_size(self):
        n = self.n_soldiers + self.n_battlefields - 1
        k = self.n_battlefields - 1
        r = min(k, n - k)
        N = reduce(op.mul, range(n, n - r, -1), 1)
        D = reduce(op.mul, range(1, r + 1), 1)
        return N // D

    def create_complete_strategy_space(self):
        size = self.strategy_space_size
        # Keep going until we put n_strategies in strategy set
        while len(self.master_dataset) < size:
            strategy = create_unique_strategy(self.n_soldiers,
                                              self.n_battlefields,
                                              self.unique_strategies)
            self.master_dataset.append(strategy)
        return self.master_dataset

    def compute_scores(self,
                       player_1_strategy,
                       player_2_strategy):
        # Determine which player wins each battlefield.
        # number_of_soldiers(Attacker) > number_of_soldiers(Defender)
        #     Attacker wins
        # Otherwise
        #     Defender wins
        player_1_score = 0
        player_2_score = 0
        # Assign scores
        for i in range(0, self.n_battlefields):
            if player_1_strategy[i] > player_2_strategy[i]:
                player_1_score += 1
            else:
                player_2_score += 1
        return player_1_score, player_2_score

    def get_strategy_score(self,
                           pl_strategy):
        # Calculates wins / losses that a strategy achieves
        # when compared against every selection in dataset
        n_wins = 0
        for strategy in self.master_dataset:
            pl_strategy_score, strategy_score = self.compute_scores(pl_strategy,
                                                                    strategy)
            # Determine if given player strategy wins, draws, or loses
            if pl_strategy_score > strategy_score:
                n_wins += 1
        return n_wins


class AttackerBot:
    def __init__(self, blotto_game, n_strategies):
        self.game = blotto_game
        # Number of learning strategies
        self.learning_strategies_count = n_strategies
        # Dictionary of unique learning strategies
        self.unique_learning_strategies = defaultdict(list)
        self.unique_strategies_count = 0
        # Initialise strategies for players randomly
        self.player_strategies = []
        self.create_learning_strategy_space()

    def create_learning_strategy_space(self):
        # Keep going until we put n_strategies in strategy set
        while len(self.player_strategies) < self.learning_strategies_count:
            strategy = create_unique_strategy(self.game.n_soldiers,
                                              self.game.n_battlefields,
                                              self.unique_learning_strategies)
            self.player_strategies.append(strategy)

    def mutate(self, strategy):
        # Mutate the given strategy slightly
        deploy_strategy = list(strategy)
        # Pick a a number in [1,..,n_bfs] for number of mutations
        n_mutations = random.randrange(self.game.n_battlefields)

        for i in range(n_mutations - 1):
            # Decrement deployment in one battlefield and increment another
            rand_idx_1 = random.randrange(self.game.n_battlefields)
            rand_idx_2 = random.randrange(self.game.n_battlefields)
            bfs_1 = deploy_strategy[rand_idx_1]
            if bfs_1 > 0:
                deploy_strategy[rand_idx_1] -= 1
                deploy_strategy[rand_idx_2] += 1

        return deploy_strategy

    def add_strategies(self, n_strategies):
        # Empty list for storing list of strategies
        l_strategies = []
        # Keep going until we put n_strategies in strategy set
        while len(l_strategies) < n_strategies:
            strategy = create_unique_strategy(self.game.n_soldiers,
                                              self.game.n_battlefields,
                                              self.unique_learning_strategies)
            l_strategies.append(strategy)
            self.player_strategies.append(strategy)

    def get_scored_strategies(self):
        strategy_scores = [(player_strategy, self.game.get_strategy_score(player_strategy))
                           for player_strategy in self.player_strategies]
        return sorted(strategy_scores, key=lambda strategy_score: strategy_score[1], reverse=True)

    def get_strategies_count(self):
        strategies_count = 0
        for key, value in self.unique_learning_strategies.items():
            strategies_count += len(value)
        return strategies_count

    def attack_add_update(self):
        # Creates the next generation of strategies from the current one
        # Maintains a list of selections, ranked by their scores. 
        # Every time this list is updated, the top third of the list is maintained, 
        # another third is composed of mutants of the top third, 
        # and the final third is composed of random selections

        # Player Strategies ranked by score
        ranked_strategies = self.get_scored_strategies()

        l_sorted_strategies = []
        for stg in ranked_strategies:
            l_sorted_strategies.append(stg[0])

        # Populate a new generation of strategies
        self.player_strategies = []

        attacker_strategies_count = self.get_strategies_count()
        total_strategies_count = self.game.strategy_space_size
        count = self.learning_strategies_count // 3
        if (total_strategies_count - attacker_strategies_count) < 2 * count:
            mutant_count = (total_strategies_count - attacker_strategies_count) // 2
            add_count = total_strategies_count - attacker_strategies_count
            count = self.learning_strategies_count - add_count
        else:
            mutant_count = count
            add_count = 2 * count

        for i in range(count):
            # Keep the top 33% player strategies from the rankings
            self.player_strategies.append(ranked_strategies[i][0])
        for k in range(mutant_count):
            # Create a mutant of the top 33% player strategies
            mutant_strategy = self.mutate(ranked_strategies[k][0])
        # Add some additional random strategies, for variety in the next generation
        self.add_strategies(add_count)

        return l_sorted_strategies


    def attack(self):
        ranked_strategies = self.get_scored_strategies()
        l_sorted_strategies = []
        for stg in ranked_strategies:
            l_sorted_strategies.append(stg[0])
        return l_sorted_strategies



print("Input the Number of Soldiers")
inp_n_soldiers = input("[User]: ")
print("Input the Number of Battlefronts")
inp_n_battlefronts = input("[User]: ")

# Instantiate Game and Bot
print("Creating Blotto Object ...\n")
blotto_game = Blotto(inp_n_soldiers, inp_n_battlefronts)
n_total_strategies = blotto_game.strategy_space_size
l_strategy_space = blotto_game.create_complete_strategy_space()

print("\nCreating AttackerBot Object")
n_learning_strategies = 60
attacker_bot = AttackerBot(blotto_game, n_learning_strategies)

l_final_strategies = []
epochs = 1000
for j in range(epochs):
    if j % 100 == 0:
        print("attacker:", attacker_bot.get_strategies_count(), "total:", n_total_strategies)
    if attacker_bot.get_strategies_count() >= n_total_strategies:
        break
    else:
        attacker_bot.attack_add_update()
l_final_strategies = attacker_bot.attack()
print("attacker:", attacker_bot.get_strategies_count(), "total:", n_total_strategies)

csv_file = "best_ev_strategies_100_100_3.csv"
with open(csv_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(l_final_strategies)


