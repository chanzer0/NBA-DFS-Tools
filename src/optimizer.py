import pulp as plp
import pandas as pd
import timeit
import cProfile
import pstats
from line_profiler import LineProfiler
pd.options.mode.chained_assignment = None  # default='warn'

# CONSTs
DK_POSITIONALITY = {'PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL'}


def add_positionality(pos_string):
    if 'G' in pos_string:
        pos_string += '/G'
    if 'F' in pos_string:
        pos_string += '/F'
    pos_string += '/UTIL'
    return pos_string


def run():
    projections = pd.read_csv('../dk_data/projections.csv',
                              thousands=',').drop_duplicates(subset='Name', keep='first')

    projections['Position'] = projections['Position'].apply(add_positionality)

    projections = projections.sort_values('Fpts', ascending=False)

    prob = plp.LpProblem("NBA_DK", plp.LpMaximize)

    # Create a list of the players
    players = list(projections['Name'])

    # Create a dictinary of salaries for players
    salaries = dict(zip(players, projections['Salary']))

    # Create a dictionary of projections for players
    fpts = dict(zip(players, projections['Fpts']))

    # Create a dictionary of positions for players
    positions = dict(zip(players, projections['Position']))
    for player in positions:
        positions[player] = {pos for pos in positions[player].split('/')}

    legal_assignments_set = {(player, pos) for player in positions.keys()
                             for pos in positions[player]}

    prob = plp.LpProblem("DK_NBA", plp.LpMaximize)

    player_vars = plp.LpVariable.dicts(
        "Player", legal_assignments_set, cat="Binary")

    # Maximize for Fpts
    prob += sum(player_vars[player, pos] * fpts[player]
                for (player, pos) in legal_assignments_set), 'Maximize Fpts Objective'

    # Salary constraint
    prob += sum(player_vars[player, pos] * salaries[player]
                for (player, pos) in legal_assignments_set) <= 50000, 'Maximize Fpts Objective'

    # Only assign each player once
    for player in players:
        prob += sum(player_vars[player, pos]
                    for pos in DK_POSITIONALITY if (player, pos) in legal_assignments_set) <= 1, f'Unique {player} Constraint'

    # Meet DK roster constraints
    for position in DK_POSITIONALITY:
        prob += sum(player_vars[player, position]
                    for player in players if (player, position) in legal_assignments_set) == 1, f'{position} Constraint'
    for x in range(20):
        print(x)
        try:
            lp = LineProfiler()
            lp_wrapper = lp(prob.solve)
            lp_wrapper(plp.PULP_CBC_CMD(msg=0))
            lp.print_stats()
        except plp.PulpSolverError:
            print('broken')
            quit()
        # print(f'Status: {plp.LpStatus[prob.status]}')

        score = str(prob.objective)
        for v in prob.variables():
            score = score.replace(v.name, str(v.varValue))

        score = eval(score)
        lineup = [v.name for v in prob.variables() if v.varValue != 0]
        # print(score, lineup)
        # Force the opto to produce a sub-optimal result by setting objective to be slightly less than the current optimal solution
        prob += sum(player_vars[player, pos] * fpts[player]
                    for (player, pos) in legal_assignments_set) <= score - 0.001, f'{score} Fpts Objective'


run()
