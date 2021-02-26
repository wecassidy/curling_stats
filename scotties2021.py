#!/usr/bin/env python3

import itertools as it
from functools import reduce
from operator import mul

import numpy as np


def validate_win_matrix(W):
    """
    Basic sanity check on the win matrix: all probabilities are in the
    right range and probability of mutually exclusive events add up to
    one.

    P(i beats i) is nonsense because teams don't play themselves, but
    diagonal entries should be 0.5 for simplicity of code.

    If the matrix passes, returns nothing. If the matrix fails, throws
    assertions with an appropriate error message.
    """
    for i in range(8):
        for j in range(8):
            assert W[i, j] + W[j, i] == 1, f"P({i} beats {j}) +  P({j} beats {i}) != 1"
            assert W[i, j] >= 0, f"P({i} beats {j}) < 0"
            assert W[i, j] <= 1, f"P({i} beats {j}) > 1"


def partition(pred, iterable):
    """
    Splits an iterable in two based on a predicate. Returns two
    iterables: false items and true items (in that order).
    """
    t1, t2 = it.tee(iterable)
    return it.filterfalse(pred, t1), filter(pred, t2)


def count(iterable):
    """Counts the elements in an iterable. Blocks until the iterable terminates."""
    return sum(1 for i in iterable)


# For convenience, the index of a team in this list is its unique
# ID. Is this a good idea? Probably not, but this is easy.
teams = [
    {"team": "ON (Homan)", "pool": "A", "pool_w": 7},
    {"team": "CA (Einarson)", "pool": "A", "pool_w": 7},
    {"team": "SK (Anderson)", "pool": "B", "pool_w": 6},
    {"team": "MB (Jones)", "pool": "B", "pool_w": 6},
    {"team": "QC (St-Georges)", "pool": "B", "pool_w": 6},
    {"team": "WC3 (Peterson)", "pool": "A", "pool_w": 5},
    {"team": "WC1 (Fleury)", "pool": "B", "pool_w": 5},
    {"team": "AB (Walker)", "pool": "A", "pool_w": 5},
]

pool_wins = np.array([team["pool_w"] for team in teams])


def game_probability(win_matrix, game, winner):
    Pab = win_matrix[game[1], game[0]]
    return Pab if game[0] == winner else 1 - Pab


def win_list(record):
    """
    Return a list of the number of wins for each team in a given
    record of games.
    """
    return np.array(
        [count(filter(lambda r: r == i, record)) for i in range(len(teams))]
    )


# win_matrix[i, j] = probability that team i beats team j
win_matrix = np.loadtxt("scotties_win_matrix_prior.csv", delimiter=",")
validate_win_matrix(win_matrix)

# Build lists of games, ways those games could go, and the probability of each sequence of wins and losses
games = tuple(
    it.product(*partition(lambda i: teams[i]["pool"] == "B", range(len(teams))))
)
records = tuple(
    tuple(game[0] if a_wins else game[1] for game, a_wins in zip(games, record))
    for record in it.product([True, False], repeat=len(games))
)
probabilities = tuple(
    reduce(
        mul,
        (
            game_probability(win_matrix, games[game], result)
            for game, result in enumerate(r)
        ),
    )
    for r in records
)


def makes_top_n(team_id, wins, n=3):
    wins_with_ids = np.array(sorted(enumerate(wins), key=lambda w: w[1], reverse=True))
    return team_id in wins_with_ids[0:n, 0]


playoff_probabilities = np.zeros((len(teams),))
for r, p in zip(records, probabilities):
    updated_wins = pool_wins + win_list(r)
    for team in range(len(teams)):
        if makes_top_n(team, updated_wins):
            playoff_probabilities[team] += p

print("Team\t\tPool W-L\tP(playoffs)")
for team, p in sorted(
    zip(teams, playoff_probabilities), key=lambda x: x[1], reverse=True
):
    wins = team["pool_w"]
    print(f"{team['team']}\t     {wins}-{8 - wins}\t{p}")
