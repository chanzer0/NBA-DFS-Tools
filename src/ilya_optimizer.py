import json
import csv
import re
import numpy as np
from pulp import *
from datetime import datetime


class NBA_Ilya_Optimizer:
    problem = None
    num_lineups = None
    player_dict = {}
    max_salary = 70.0
    lineups = {}
    tournament_id = None

    def __init__(self, num_lineups, use_rand):
        self.problem = LpProblem('NBA', LpMaximize)
        self.load_projections('ilya_projections.csv')
        self.load_boom_bust('boom_bust.csv')
        self.num_lineups = num_lineups
        self.use_rand = True if use_rand == 'rand' else False

    # Load config from file
    def load_config(self):
        with open('config.json') as json_file: 
            self.config = json.load(json_file) 

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

                self.player_dict[player_name]['ID'] = row['Player ID']

                #some players have 2 positions - will be listed like 'PG/SF' or 'PF/C'
                self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]

                # need to pre-emptively set ownership to 0 as some players will not have ownership
                # if a player does have ownership, it will be updated later on in load_ownership()
                self.player_dict[player_name]['Salary'] = float(row['Salary'])

                if self.tournament_id is None:
                    self.tournament_id = int(row['Tournament ID'])
                
    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot. 
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # lp_variables = {'PG': 'LeBron James' : 'PG_LeBron James', 'Kyrie Irving': 'PG_Kyrie_Irving', .... , 'SG' : .... , etc... }
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        if self.use_rand:
            print('use rand')
            self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict)
        else:
            self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict)
        
        # Set the salary constraints
        self.problem += lpSum(self.player_dict[player]['Salary'] * lp_variables[player] for player in self.player_dict) <= self.max_salary

        # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'guard_point' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'guard_point' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 shooting guard, can have up to 3 if utilizing G and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'guard_shooting' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'guard_shooting' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 small forward, can have up to 3 if utilizing F and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'forward_small' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'forward_small' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 power forward, can have up to 3 if utilizing F and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'forward_power' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'forward_power' in self.player_dict[player]['Position']) <= 3
        # Need at least 1 center, can have up to 2 if utilizing C and UTIL slots
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'center' in self.player_dict[player]['Position']) >= 1
        self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'center' in self.player_dict[player]['Position']) <= 3
        # Can only roster 8 total players
        self.problem += lpSum(lp_variables[player] for player in self.player_dict) == 7

        # Crunch!
        for i in range(int(self.num_lineups)):
            self.problem.solve(PULP_CBC_CMD(msg=0))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            print(i)
            player_names = [v.name.replace('_', ' ') for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names

            # Dont generate the same lineup twice
            if self.use_rand:
                self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict)
            else:
                self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict) <= (fpts - 0.01)
           

    def output(self):
        self.format_lineups()
        with open('optimal_lineups_{}.csv'.format(datetime.now().strftime("%d-%m-%Y_%H%M%S")), 'w') as f:
            f.write('guard_point,guard_shooting,forward_small,forward_power,center,flex,flex,Fpts Projection,Salary,\n')
            for fpts, x in self.lineups.items():
                salary = sum(self.player_dict[player]['Salary'] for player in x)
                lineup_str = '{},{},{},{},{},{},{},{},{}'.format(
                    x[0].replace('#', '-'),
                    x[1].replace('#', '-'),
                    x[2].replace('#', '-'),
                    x[3].replace('#', '-'),
                    x[4].replace('#', '-'),
                    x[5].replace('#', '-'),
                    x[6].replace('#', '-'),
                    round(fpts, 2),round(salary, 2)
                )
                f.write('%s\n' % lineup_str)
        with open('optimal_lineups_upload_{}.csv'.format(datetime.now().strftime("%d-%m-%Y_%H%M%S")), 'w') as f:
            f.write('tournament_id,guard_point,guard_shooting,forward_small,forward_power,center,flex,flex\n')
            for fpts, x in self.lineups.items():
                salary = sum(self.player_dict[player]['Salary'] for player in x)
                lineup_str = '{},{},{},{},{},{},{},{}'.format(
                    self.tournament_id,
                    self.player_dict[x[0]]['ID'],
                    self.player_dict[x[1]]['ID'],
                    self.player_dict[x[2]]['ID'],
                    self.player_dict[x[3]]['ID'],
                    self.player_dict[x[4]]['ID'],
                    self.player_dict[x[5]]['ID'],
                    self.player_dict[x[6]]['ID']
                )
                f.write('%s\n' % lineup_str)
        with open('player_exposure_{}.csv'.format(datetime.now().strftime("%d-%m-%Y_%H%M%S")), 'w') as f:
            f.write('player_name,in,total,exposure%')
            players = set(x for l in self.lineups.values() for x in l)
            total = len(self.lineups.values())
            for player in players:
                lineups_in = sum([lineup.count(player) for lineup in self.lineups.values()])
                exposure = round(lineups_in/total * 100, 2)
                exposure_str = '{},{},{},{}%'.format(player.replace('#','-'),lineups_in,total,exposure)
                f.write('%s\n' % exposure_str)

    def format_lineups(self):
        temp = self.lineups.items()
        self.lineups = {}
        finalized = [None] * 7
        for fpts,lineup in temp:
            for player in lineup:
                if 'guard_point' in self.player_dict[player]['Position']:
                    if finalized[0] is None:
                        finalized[0] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'guard_shooting' in self.player_dict[player]['Position']:
                    if finalized[1] is None:
                        finalized[1] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'forward_small' in self.player_dict[player]['Position']:
                    if finalized[2] is None:
                        finalized[2] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'forward_power' in self.player_dict[player]['Position']:
                    if finalized[3] is None:
                        finalized[3] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

                elif 'center' in self.player_dict[player]['Position']:
                    if finalized[4] is None:
                        finalized[4] = player
                    elif finalized[5] is None:
                        finalized[5] = player
                    else:
                        finalized[6] = player

            self.lineups[fpts] = finalized
            finalized = [None] * 7

def main(num_lineups, use_rand):
    opto = NBA_Ilya_Optimizer(num_lineups, use_rand)
    opto.optimize()
    opto.output()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], None)