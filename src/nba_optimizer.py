import json
import csv
import os
import datetime
import pytz
import timedelta
import numpy as np
import pulp as plp
from itertools import groupby
from random import shuffle, choice


class NBA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    use_randomness = None
    team_list = []
    lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    global_team_limit = None

    def __init__(self, site=None, num_lineups=0, use_randomness=False, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.use_randomness = use_randomness == 'rand'
        self.load_config()
        self.problem = plp.LpProblem('NBA', plp.LpMaximize)

        projection_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        ownership_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        self.load_ownership(ownership_path)

        player_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        boom_bust_path = os.path.join(os.path.dirname(
            __file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        self.load_rules()

    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:
                    if self.site == 'dk':
                        self.player_dict[player_name]['ID'] = int(row['ID'])
                    else:
                        self.player_dict[player_name]['ID'] = row['Id']

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = self.config["global_team_limit"]

    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    # if this player has a very low chance of reaching a GPP target score, do not play them
                    if float(row['Boom%']) < 0.0:
                        del self.player_dict[player_name]
                        continue
                    self.player_dict[player_name]['Leverage'] = float(
                        row['Leverage'])
                    self.player_dict[player_name]['StdDev'] = float(
                        row['Std Dev'])
                    self.player_dict[player_name]['Boom'] = float(row['Boom%'])
                    self.player_dict[player_name]['Bust'] = float(row['Bust%'])
                    self.player_dict[player_name]['Ceiling'] = float(
                        row['Ceiling'])
                    if row['Optimal%'] != '':
                        self.player_dict[player_name]['Optimal'] = float(
                            row['Optimal%'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if float(row['Fpts']) < 0:
                    continue
                self.player_dict[player_name] = {'Fpts': 0, 'Position': None, 'ID': 0, 'Salary': 0, 'Name': '',
                                                 'StdDev': 0, 'Team': '', 'Ownership': 0.1, 'Optimal': 0, 'Minutes': 0, 'Boom': 0, 'Bust': 0, 'Ceiling': 0}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(
                    row['Salary'].replace(',', ''))
                self.player_dict[player_name]['Minutes'] = float(
                    row['Minutes'])
                self.player_dict[player_name]['Name'] = row['Name']
                self.player_dict[player_name]['Team'] = row['Team']

                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])

                # Need to handle MPE
                self.player_dict[player_name]['Position'] = [
                    pos for pos in row['Position'].split('/')]

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(
                        row['Ownership %'])

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {player: plp.LpVariable(
            player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        if self.use_randomness:
            self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'], (self.player_dict[player]
                                      ['StdDev'])) * lp_variables[player] for player in self.player_dict), 'Objective'
        else:
            self.problem += plp.lpSum(self.player_dict[player]['Fpts'] * lp_variables[player]
                                      for player in self.player_dict), 'Objective'

        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 45000 if self.site == 'dk' else 55000
        self.problem += plp.lpSum(self.player_dict[player]['Salary'] *
                                  lp_variables[player] for player in self.player_dict) <= max_salary
        self.problem += plp.lpSum(self.player_dict[player]['Salary'] *
                                  lp_variables[player] for player in self.player_dict) >= min_salary

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                          for player in group) >= int(limit)
        for limit, groups in self.at_most.items():
            for group in groups:
                self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                          for player in group) <= int(limit)

        # Address team limits
        for team, limit in self.team_limits.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(limit)
        if self.global_team_limit is not None:
            for team in self.team_list:
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict if self.player_dict[player]['Team'] == team) <= int(self.global_team_limit)

        if self.site == 'dk':
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 shooting guard, can have up to 3 if utilizing G and UTIL slots
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 small forward, can have up to 3 if utilizing F and UTIL slots
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) <= 3
            # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'C' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'C' in self.player_dict[player]['Position']) <= 2
            # Need at least 3 guards (PG,SG,G)
            self.problem += plp.lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]
                                      ['Position'] or 'SG' in self.player_dict[player]['Position']) >= 3
            # Need at least 3 forwards (SF,PF,F)
            self.problem += plp.lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]
                                      ['Position'] or 'PF' in self.player_dict[player]['Position']) >= 3
            # Max 4 of PG and C single-eligbility players to prevent rare edge case where you can end up with 3 PG and 2 C, even though this build is infeasible
            self.problem += plp.lpSum(lp_variables[player] for player in self.player_dict if [
                                      'PG'] == self.player_dict[player]['Position'] or ['C'] == self.player_dict[player]['Position']) <= 4
            # Can only roster 8 total players
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict) == 8
        else:
            # PG MPE
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) >= 2
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) <= 5
            # SG MPE
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) >= 2
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) <= 5
            # SF MPE
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) >= 2
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) <= 5
            # PF MPE
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) >= 2
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) <= 5
            # C MPE
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'C' in self.player_dict[player]['Position']) >= 1
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if 'C' in self.player_dict[player]['Position']) <= 3

            # PG Alignment
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if ['PG'] == self.player_dict[player]['Position']) <= 2

            # SG Alignment
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if ['SG'] == self.player_dict[player]['Position']) <= 2

            # SF Alignment
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if ['SF'] == self.player_dict[player]['Position']) <= 2

            # PF Alignment
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if ['PF'] == self.player_dict[player]['Position']) <= 2

            # C Alignment
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict if ['C'] == self.player_dict[player]['Position']) <= 1

            # Can only roster 9 total players
            self.problem += plp.lpSum(lp_variables[player]
                                      for player in self.player_dict) == 9
            # Max 4 per team
            for team in self.team_list:
                self.problem += plp.lpSum(lp_variables[player]
                                          for player in self.player_dict if self.player_dict[player]['Team'] == team) <= 4

        # Crunch!
        for i in range(self.num_lineups):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(
                    len(self.num_lineups), self.num_lineups))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            if i % 100 == 0:
                print(i)
            player_names = [v.name.replace(
                '_', ' ') for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names

            # Dont generate the same lineup twice
            if self.use_randomness:
                # Set a new random fpts projection within their distribution
                self.problem += plp.lpSum(np.random.normal(
                    self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict)
            else:
                # Enforce this by lowering the objective i.e. producing sub-optimal results
                self.problem += plp.lpSum(self.player_dict[player]['Fpts'] * lp_variables[player]
                                          for player in self.player_dict) <= (fpts - 0.01)

    def output(self):
        print('Lineups done generating. Outputting.')
        unique = {}
        for fpts, lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        self.lineups = unique
        if self.num_uniques != 1:
            num_uniq_lineups = plp.OrderedDict(
                sorted(self.lineups.items(), reverse=False, key=lambda t: t[0]))
            self.lineups = {}
            for fpts, lineup in num_uniq_lineups.copy().items():
                temp_lineups = list(num_uniq_lineups.values())
                temp_lineups.remove(lineup)
                use_lineup = True
                for x in temp_lineups:
                    common_players = set(x) & set(lineup)
                    roster_size = 9 if self.site == 'fd' else 8
                    if (roster_size - len(common_players)) < self.num_uniques:
                        use_lineup = False
                        del num_uniq_lineups[fpts]
                        break

                if use_lineup:
                    self.lineups[fpts] = lineup
        self.format_lineups()

        out_path = os.path.join(os.path.dirname(
            __file__), '../output/{}_optimal_lineups.csv'.format(self.site))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write(
                    'PG,SG,SF,PF,C,G,F,UTIL,Salary,Fpts Proj,Ceiling,Own. Product,Optimal%,Minutes,Boom%,Bust%,Leverage\n')
                for fpts, x in self.lineups.items():
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    ceil = sum(self.player_dict[player]
                               ['Ceiling'] for player in x)
                    mins = sum(self.player_dict[player]
                               ['Minutes'] for player in x)
                    boom_p = np.prod(
                        [self.player_dict[player]['Boom'] for player in x])
                    bust_p = np.prod(
                        [self.player_dict[player]['Bust'] for player in x])
                    optimal_p = np.prod(
                        [self.player_dict[player]['Optimal'] for player in x])
                    leverage = sum(self.player_dict[player]
                                   ['Leverage'] for player in x)
                    # print(sum(self.player_dict[player]['Ownership'] for player in x))
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{},{},{}'.format(
                        x[0].replace('#', '-'), self.player_dict[x[0]]['ID'],
                        x[1].replace('#', '-'), self.player_dict[x[1]]['ID'],
                        x[2].replace('#', '-'), self.player_dict[x[2]]['ID'],
                        x[3].replace('#', '-'), self.player_dict[x[3]]['ID'],
                        x[4].replace('#', '-'), self.player_dict[x[4]]['ID'],
                        x[5].replace('#', '-'), self.player_dict[x[5]]['ID'],
                        x[6].replace('#', '-'), self.player_dict[x[6]]['ID'],
                        x[7].replace('#', '-'), self.player_dict[x[7]]['ID'],
                        salary, round(fpts_p, 2), round(
                            ceil, 2), own_p, optimal_p, mins, boom_p, bust_p, leverage
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write(
                    'PG,PG,SG,SG,SF,SF,PF,PF,C,Salary,Fpts Proj,Ceiling,Own. Product,Optimal%,Minutes,Boom%,Bust%,Leverage\n')
                for fpts, x in self.lineups.items():
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    ceil = sum(self.player_dict[player]
                               ['Ceiling'] for player in x)
                    mins = sum(self.player_dict[player]
                               ['Minutes'] for player in x)
                    boom = np.prod(
                        [self.player_dict[player]['Boom'] for player in x])
                    bust = np.prod(
                        [self.player_dict[player]['Bust'] for player in x])
                    optimal = np.prod(
                        [self.player_dict[player]['Optimal'] for player in x])
                    leverage = sum(self.player_dict[player]
                                   ['Leverage'] for player in x)
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['ID'], x[0].replace('#', '-'),
                        self.player_dict[x[1]]['ID'], x[1].replace('#', '-'),
                        self.player_dict[x[2]]['ID'], x[2].replace('#', '-'),
                        self.player_dict[x[3]]['ID'], x[3].replace('#', '-'),
                        self.player_dict[x[4]]['ID'], x[4].replace('#', '-'),
                        self.player_dict[x[5]]['ID'], x[5].replace('#', '-'),
                        self.player_dict[x[6]]['ID'], x[6].replace('#', '-'),
                        self.player_dict[x[7]]['ID'], x[7].replace('#', '-'),
                        self.player_dict[x[8]]['ID'], x[8].replace('#', '-'),
                        salary, round(fpts_p, 2), round(
                            ceil, 2), own_p, optimal, mins, boom, bust, leverage
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')

    def format_lineups(self):
        if self.site == 'dk':
            dk_roster = [['PG'], ['SG'], ['SF'], ['PF'], ['C'], [
                'PG', 'SG'], ['SF', 'PF'], ['PG', 'SG', 'SF', 'PF', 'C']]
            temp = self.lineups.items()
            self.lineups = {}
            for fpts, lineup in temp:
                finalized = [None] * 8
                z = 0
                cond = False
                while None in finalized:
                    if cond:
                        break
                    indices = [0, 1, 2, 3, 4, 5, 6, 7]
                    shuffle(indices)
                    for i in indices:
                        if finalized[i] is None:
                            eligible_players = []
                            for player in lineup:
                                if any(pos in dk_roster[i] for pos in self.player_dict[player]['Position']):
                                    eligible_players.append(player)
                            selected = choice(eligible_players)
                            # if there is an eligible player for this position not already in the finalized roster
                            if any(player not in finalized for player in eligible_players):
                                while selected in finalized:
                                    selected = choice(eligible_players)
                                finalized[i] = selected
                            # this lineup combination is no longer feasible - retry
                            else:
                                z += 1
                                if z == 1000:
                                    cond = True
                                    break

                                shuffle(indices)
                                finalized = [None] * 8
                                break
                if not cond:
                    self.lineups[fpts] = finalized
        else:
            fd_roster = [['PG'], ['PG'], ['SG'], ['SG'], ['SF'], [
                'SF'], ['PF'], ['PF'], ['C']]
            temp = self.lineups.items()
            self.lineups = {}
            for fpts, lineup in temp:
                finalized = [None] * 9
                z = 0
                cond = False
                infeasible = False
                while None in finalized:
                    if cond:
                        break
                    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    shuffle(indices)
                    for i in indices:
                        if finalized[i] is None:
                            eligible_players = []
                            for player in lineup:
                                if any(pos in fd_roster[i] for pos in self.player_dict[player]['Position']):
                                    eligible_players.append(player)
                            selected = choice(eligible_players)
                            # if there is an eligible player for this position not already in the finalized roster
                            if any(player not in finalized for player in eligible_players):
                                while selected in finalized:
                                    selected = choice(eligible_players)
                                finalized[i] = selected
                            # this lineup combination is no longer feasible - retry
                            else:
                                z += 1
                                if z == 1000:
                                    #print('infeasible lineup')
                                    # print(lineup)
                                    # for player in lineup:
                                    #     if player is not None:
                                    #         print(
                                    #             player, self.player_dict[player]['Fpts'], self.player_dict[player]['Position'])
                                    #     else:
                                    #         print(player)
                                    cond = True
                                    infeasible = True
                                    break

                                shuffle(indices)
                                finalized = [None] * 9
                                break
                if not cond and not infeasible:
                    self.lineups[fpts] = finalized
