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
    team_list = []
    lineups = {}
    player_dict = {}
    at_least = {}
    at_most = {}
    team_limits = {}
    matchup_limits = {}
    matchup_at_least = {}
    global_team_limit = None
    projection_minimum = 0
    randomness_amount = 0

    def __init__(self, site=None, num_lineups=0, num_uniques=1):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.load_config()
        self.load_rules()

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

    # Load config from file

    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json'), encoding='utf-8-sig') as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name_key = 'Name' if self.site == 'dk' else 'Nickname'
                player_name = row[name_key].replace('-', '#')
                if player_name in self.player_dict:

                    if self.site == 'dk':
                        self.player_dict[player_name]['RealID'] = int(
                            row['ID'])
                        self.player_dict[player_name]['ID'] = int(
                            row['ID'][-3:])
                        self.player_dict[player_name]['Matchup'] = row['Game Info'].split(' ')[
                            0]
                    else:
                        self.player_dict[player_name]['RealID'] = str(
                            row['Id'])
                        self.player_dict[player_name]['ID'] = int(
                            row['Id'].split('-')[1])
                        self.player_dict[player_name]['Matchup'] = row['Game'].split(' ')[
                            0]

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.matchup_limits = self.config["matchup_limits"]
        self.matchup_at_least = self.config["matchup_at_least"]

    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path, encoding='utf-8-sig') as file:
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
                        row['Projection']) + float(row['Std Dev'])
                    if row['Optimal%'] != '':
                        self.player_dict[player_name]['Optimal'] = float(
                            row['Optimal%'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if float(row['Fpts']) < self.projection_minimum:
                    continue
                self.player_dict[player_name] = {'Fpts': 0.1, 'Position': None, 'ID': 0, 'Salary': 1000, 'Name': '', 'RealID': 0, 'Matchup': '', 'Leverage': 0,
                                                 'StdDev': 0.1, 'Team': '', 'Ownership': 0.1, 'Optimal': 0.1, 'Minutes': 1, 'Boom': 0.1, 'Bust': 0.1, 'Ceiling': 0.1}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(
                    row['Salary'].replace(',', ''))
                self.player_dict[player_name]['Minutes'] = float(
                    row['Minutes'])
                self.player_dict[player_name]['Name'] = row['Name']
                self.player_dict[player_name]['Team'] = row['Team']
                self.player_dict[player_name]['FPPM'] = float(
                    row['Fpts']) / float(row['Minutes'])
                self.player_dict[player_name]['PPD'] = float(
                    row['Fpts']) / int(row['Salary'].replace(',', ''))*1000
                if row['Team'] not in self.team_list:
                    self.team_list.append(row['Team'])

                # print(self.player_dict[player_name])

                # Need to handle MPE
                self.player_dict[player_name]['Position'] = [
                    pos for pos in row['Position'].split('/')]

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path, encoding='utf-8-sig') as file:
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

        # set the objective - maximize fpts & set randomness amount from config
        self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'],
                                                   (self.player_dict[player]['StdDev'] * self.randomness_amount / 100))
                                  * lp_variables[player] for player in self.player_dict), 'Objective'
        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        min_salary = 49000 if self.site == 'dk' else 59000
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

        for matchup, limit in self.matchup_limits.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) <= int(limit)

        for matchup, limit in self.matchup_at_least.items():
            self.problem += plp.lpSum(lp_variables[player.replace('-', '#')]
                                      for player in self.player_dict if self.player_dict[player]['Matchup'] == matchup) >= int(limit)

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

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += plp.lpSum(np.random.normal(self.player_dict[player]['Fpts'],
                                                           (self.player_dict[player]['StdDev'] * self.randomness_amount / 100))
                                          * lp_variables[player] for player in self.player_dict), 'Objective'
            else:
                self.problem += plp.lpSum(self.player_dict[player]['Fpts'] * lp_variables[player]
                                          for player in self.player_dict) <= (fpts - 0.001)

    def output(self):
        print('Lineups done generating. Outputting.')
        unique = {}
        for fpts, lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        # for fpts, lineup in self.lineups.items():
        #     id_sum = sum(self.player_dict[player]['ID'] for player in lineup)
        #     print(id_sum, lineup)
        #     if lineup not in unique.values():
        #         unique[id_sum] = (fpts, lineup)

        # { 'id_sum': ('fpts', 'lineup')}
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
                    'PG,SG,SF,PF,C,G,F,UTIL,Salary,Fpts Proj,Ceiling,Own. Product,Optimal%,Minutes,Boom%,Bust%,Leverage,FPPM,PPD,STDDEV\n')
                for fpts, x in self.lineups.items():
                    # print(id_sum, tple)
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    ceil = sum([self.player_dict[player]
                               ['Ceiling'] for player in x])
                    mins = sum([self.player_dict[player]
                               ['Minutes'] for player in x])
                    boom_p = np.prod(
                        [self.player_dict[player]['Boom'] for player in x])
                    bust_p = np.prod(
                        [self.player_dict[player]['Bust'] for player in x])
                    optimal_p = np.prod(
                        [self.player_dict[player]['Optimal'] for player in x])
                    leverage = sum(self.player_dict[player]
                                   ['Leverage'] for player in x)
                    fppm = np.prod(
                        [self.player_dict[player]['FPPM'] for player in x])
                    ppd = np.prod(
                        [self.player_dict[player]['PPD'] for player in x])
                    stddev = np.prod(
                        [self.player_dict[player]['StdDev'] for player in x])
                    # print(sum(self.player_dict[player]['Ownership'] for player in x))
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                        x[0].replace(
                            '#', '-'), self.player_dict[x[0]]['RealID'],
                        x[1].replace(
                            '#', '-'), self.player_dict[x[1]]['RealID'],
                        x[2].replace(
                            '#', '-'), self.player_dict[x[2]]['RealID'],
                        x[3].replace(
                            '#', '-'), self.player_dict[x[3]]['RealID'],
                        x[4].replace(
                            '#', '-'), self.player_dict[x[4]]['RealID'],
                        x[5].replace(
                            '#', '-'), self.player_dict[x[5]]['RealID'],
                        x[6].replace(
                            '#', '-'), self.player_dict[x[6]]['RealID'],
                        x[7].replace(
                            '#', '-'), self.player_dict[x[7]]['RealID'],
                        salary, round(
                            fpts_p, 2), ceil, own_p, optimal_p, mins, boom_p, bust_p, leverage, fppm, ppd, stddev
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write(
                    'PG,PG,SG,SG,SF,SF,PF,PF,C,Salary,Fpts Proj,Ceiling,Own. Product,Optimal%,Minutes,Boom%,Bust%,Leverage,FPPM,PPD,STDDEV\n')
                for fpts, x in self.lineups.items():
                    salary = sum(
                        self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(
                        self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod(
                        [self.player_dict[player]['Ownership'] for player in x])
                    ceil = np.prod([self.player_dict[player]
                                   ['Ceiling'] for player in x])
                    mins = np.prod([self.player_dict[player]
                                   ['Minutes'] for player in x])
                    boom = np.prod(
                        [self.player_dict[player]['Boom'] for player in x])
                    bust = np.prod(
                        [self.player_dict[player]['Bust'] for player in x])
                    optimal = np.prod(
                        [self.player_dict[player]['Optimal'] for player in x])
                    fppm = np.prod(
                        [self.player_dict[player]['FPPM'] for player in x])
                    ppd = np.prod(
                        [self.player_dict[player]['PPD'] for player in x])
                    stddev = np.prod(
                        [self.player_dict[player]['StdDev'] for player in x])
                    leverage = sum(self.player_dict[player]
                                   ['Leverage'] for player in x)
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['RealID'], x[0].replace(
                            '#', '-'),
                        self.player_dict[x[1]]['RealID'], x[1].replace(
                            '#', '-'),
                        self.player_dict[x[2]]['RealID'], x[2].replace(
                            '#', '-'),
                        self.player_dict[x[3]]['RealID'], x[3].replace(
                            '#', '-'),
                        self.player_dict[x[4]]['RealID'], x[4].replace(
                            '#', '-'),
                        self.player_dict[x[5]]['RealID'], x[5].replace(
                            '#', '-'),
                        self.player_dict[x[6]]['RealID'], x[6].replace(
                            '#', '-'),
                        self.player_dict[x[7]]['RealID'], x[7].replace(
                            '#', '-'),
                        self.player_dict[x[8]]['RealID'], x[8].replace(
                            '#', '-'),
                        salary, round(
                            fpts_p, 2), ceil, own_p, optimal, mins, boom, bust, leverage, fppm, ppd, stddev
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
                            # print(lineup, dk_roster[i], eligible_players)
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
                                    # print('infeasible lineup')
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
