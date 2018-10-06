import csv


def compute_scores(n_battlefields,
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
    for i in range(0, n_battlefields):
        if player_1_strategy[i] > player_2_strategy[i]:
            player_1_score += 1
        else:
            player_2_score += 1
    return player_1_score, player_2_score


def get_strategy_score(n_battlefields,
                       attacker_strategy,
                       defender_list):
    # Calculates wins / losses that a strategy achieves
    # when compared against every selection in dataset
    n_wins = 0
    for defender_strategy in defender_list:
        attacker_score, defender_score = compute_scores(n_battlefields,
                                                        attacker_strategy,
                                                        defender_strategy)
        # Determine if given player strategy wins or loses
        if attacker_score > defender_score:
            n_wins += 1
    return n_wins


csv_file_1 = "best_ev_strategies_100_100_3.csv"
csv_file_2 = "best_lp_strategies_100_100_3.csv"

with open(csv_file_1, "r") as f1:
    ev_strategies_list = [list(map(int, rec)) for rec in csv.reader(f1, delimiter=',')]

with open(csv_file_2, "r") as f2:
    lp_strategies_list = [list(map(int, rec)) for rec in csv.reader(f2, delimiter=',')]

# Evolutionary strategies List -- Attacker
# LP Defender Strategies List -- Defender
# Playing Round Robin against each other
n_bfs = 3
scored_strategy_list = []
winning_strategy_list = []

print("Attacker: Evolutionary Strategies")
print("Defender: LP Strategies")
print("Total No Of Battles: 60")
print("=================================")
for a_strategy in ev_strategies_list:
    a_score = get_strategy_score(n_bfs,
                                 a_strategy,
                                 lp_strategies_list)
    if a_score > 30:
        scored_strategy_list.append((a_strategy, a_score))

winning_strategy_list = sorted(scored_strategy_list,
                               key=lambda strategy_score: strategy_score[1],
                               reverse=True)
for stg in winning_strategy_list:
    print("Strategy:", stg[0], "No Of Battles Won:", stg[1])
print("Total War Wins:", len(winning_strategy_list))

scored_strategy_list = []
winning_strategy_list = []
print("\nAttacker: LP Strategies")
print("Defender: Evolutionary Strategies")
print("Total No Of Battles: 60")
print("=================================")
for a_strategy in lp_strategies_list:
    a_score = get_strategy_score(n_bfs,
                                 a_strategy,
                                 ev_strategies_list)
    if a_score > 30:
        scored_strategy_list.append((a_strategy, a_score))

winning_strategy_list = sorted(scored_strategy_list,
                               key=lambda strategy_score: strategy_score[1],
                               reverse=True)
for stg in winning_strategy_list:
    print("Strategy:", stg[0], "No Of Battles Won:", stg[1])
print("Total War Wins:", len(winning_strategy_list))