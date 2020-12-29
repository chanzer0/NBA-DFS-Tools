import json
import csv
import re
import numpy as np
from pulp import *


class FD_NBA_Optimizer:
    problem = None
    config = None
    output_filepath = None
    num_lineups = None
    player_dict = {}
    roster_construction = {'PG': 2, 'SG': 2, 'SF': 2, 'PF': 2, 'C': 1}
    max_salary = 60000
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
                player_name = row['Nickname'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['ID'] = row['Id']

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
                self.player_dict[player_name] = {'Fpts': 0, 'Position': [], 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])

                #some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                self.player_dict[player_name]['Position'] = row['Position']

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

        # Need 2 PG
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PG' == self.player_dict[player]['Position']) == 2
        
        # Need 2 SG
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SG' == self.player_dict[player]['Position']) == 2

        # Need 2 SF
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'SF' == self.player_dict[player]['Position']) == 2

        # Need 2 PF
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PF' == self.player_dict[player]['Position']) == 2

        # Need 1 center
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'C' == self.player_dict[player]['Position']) == 1
        
        # Can only roster 9 total players
        self.problem += lpSum(lp_variables[player] for player in self.player_dict) == 9

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
        self.format_lineups()

        unique = {}
        for fpts,lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        with open(self.output_filepath, 'w') as f:
            f.write('PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Fpts Sim,Salary,Own. Product\n')
            for fpts, x in unique.items():
                salary = sum(self.player_dict[player]['Salary'] for player in x)
                fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
                own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
                lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{}'.format(
                    self.player_dict[x[0]]['ID'],x[0].replace('#', '-'),
                    self.player_dict[x[1]]['ID'],x[1].replace('#', '-'),
                    self.player_dict[x[2]]['ID'],x[2].replace('#', '-'),
                    self.player_dict[x[3]]['ID'],x[3].replace('#', '-'),
                    self.player_dict[x[4]]['ID'],x[4].replace('#', '-'),
                    self.player_dict[x[5]]['ID'],x[5].replace('#', '-'),
                    self.player_dict[x[6]]['ID'],x[6].replace('#', '-'),
                    self.player_dict[x[7]]['ID'],x[7].replace('#', '-'),
                    self.player_dict[x[8]]['ID'],x[8].replace('#', '-'),
                    round(fpts_p, 2),round(fpts, 2),salary,own_p
                )
                f.write('%s\n' % lineup_str)

    def format_lineups(self):
        temp = self.lineups.items()
        self.lineups = {}
        finalized = [None] * 9
        for fpts,lineup in temp:
            for player in lineup:
                if 'PG' == self.player_dict[player]['Position']:
                    if finalized[0] is None:
                        finalized[0] = player
                    else:
                        finalized[1] = player

                elif 'SG' == self.player_dict[player]['Position']:
                    if finalized[2] is None:
                        finalized[2] = player
                    else:
                        finalized[3] = player

                elif 'SF' == self.player_dict[player]['Position']:
                    if finalized[4] is None:
                        finalized[4] = player
                    else:
                        finalized[5] = player

                elif 'PF' == self.player_dict[player]['Position']:
                    if finalized[6] is None:
                        finalized[6] = player
                    else:
                        finalized[7] = player

                else:
                    finalized[8] = player

            self.lineups[fpts] = finalized
            finalized = [None] * 9