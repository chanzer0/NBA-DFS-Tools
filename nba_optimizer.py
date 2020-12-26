import json
import csv
import re
import numpy as np
from pulp import *


class NBA_Optimizer:
    problem = None
    config = None
    output_filepath = None
    num_lineups = None
    player_dict = {}
    player_positions = {'PG': {}, 'SG': {}, 'SF': {}, 'PF': {}, 'C': {}, 'F': {}, 'G': {}, 'UTIL': {}}
    roster_construction = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'F': 1, 'G': 1, 'UTIL': 1}
    max_salary = 50000
    lineups = {}

    def __init__(self):
        self.problem = LpProblem('NBA', LpMaximize)
        self.load_config()
        self.load_projections(self.config['projection_path'])
        self.load_ownership(self.config['ownership_path'])
        self.load_player_ids(self.config['player_path'])
        self.load_boom_bust(self.config['boombust_path'])
        self.output_filepath = self.config['output_path']
        self.num_lineups = self.config['num_lineups']

    # Load config from file
    def load_config(self):
        with open('config.json') as json_file: 
            self.config = json.load(json_file) 

    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['ID'] = int(row['ID'])

    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                self.player_dict[player_name] = {'Fpts': 0, 'Position': [], 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])

                #some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]

                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))
                # need to pre-emptively set ownership to 0 as some players will not have ownership
                # if a player does have ownership, it will be updated later on in load_ownership()
                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))
                
    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])

    def optimize(self, use_randomness):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot. 
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # lp_variables = {'PG': 'LeBron James' : 'PG_LeBron James', 'Kyrie Irving': 'PG_Kyrie_Irving', .... , 'SG' : .... , etc... }
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        if use_randomness == 'rand':
            self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict), 'Objective'
        else:
            self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict), 'Objective'

        # Set the salary constraints
        self.problem += lpSum(self.player_dict[player]['Salary'] * lp_variables[player] for player in self.player_dict) <= self.max_salary

        # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 shooting guard, can have up to 3 if utilizing G and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 small forward, can have up to 3 if utilizing F and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' in self.player_dict[player]['Position']) <= 2
        # Can only roster 8 total players
        self.problem += lpSum(lp_variables[player] for player in self.player_dict) == 8

        # Crunch!
        for i in range(self.num_lineups):
            self.problem.solve(PULP_CBC_CMD(msg=0))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            if i % 100 == 0:
                print(i)

            player_names = [v.name.replace('_', ' ') for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names

            # Dont generate the same lineup twice
            if use_randomness == 'rand':
                # Enforce this by lowering the objective i.e. producing sub-optimal results
                self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev'])* lp_variables[player] for player in self.player_dict)
            else:
                # Set a new random fpts projection within their distribution
                self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict) <= (fpts - 0.01)

    def output(self):
        unique = {}
        for fpts, x in self.lineups.items():
            salary = sum(self.player_dict[player]['Salary'] for player in x)
            fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
            own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
            lineup_str = '{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                x[0].replace('#', '-'),x[1].replace('#', '-'),
                x[2].replace('#', '-'),x[3].replace('#', '-'),
                x[4].replace('#', '-'),x[5].replace('#', '-'),
                x[6].replace('#', '-'),x[7].replace('#', '-'),
                round(fpts_p, 2),round(fpts, 2),salary,own_p
            )
            unique[round(fpts_p, 2)] = lineup_str

        with open(self.output_filepath, 'w') as f:
            f.write('PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Fpts Sim,Salary,Own. Product\n')
            for fpts, lineup_str in unique.items():
                f.write('%s\n' % lineup_str)

    def format_lineups(self):
        temp = self.lineups.items()
        self.lineups = {}
        finalized = [None] * 8
        for fpts,lineup in temp:
            for player in lineup:
                if 'PG' in self.player_dict[player]['Position']:
                    if finalized[0] is None:
                        finalized[0] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'SG' in self.player_dict[player]['Position']:
                    if finalized[1] is None:
                        finalized[1] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'SF' in self.player_dict[player]['Position']:
                    if finalized[2] is None:
                        finalized[2] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'PF' in self.player_dict[player]['Position']:
                    if finalized[3] is None:
                        finalized[3] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'C' in self.player_dict[player]['Position']:
                    if finalized[4] is None:
                        finalized[4] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

            self.lineups[fpts] = finalized
            finalized = [None] * 7