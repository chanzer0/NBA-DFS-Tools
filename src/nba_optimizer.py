import json, csv, os
import numpy as np
from pulp import *
from itertools import groupby

class NBA_Optimizer:
    site = None
    config = None
    problem = None
    output_dir = None
    num_lineups = None
    num_uniques = None
    use_randomness = None
    lineups = {}
    unique_dict = {}
    player_dict = {}

    def __init__(self, site, num_lineups, use_randomness, num_uniques):
        self.site = site
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.use_randomness = use_randomness == 'rand'
        self.load_config()
        self.problem = LpProblem('NBA', LpMaximize)

        projection_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['projection_path']))
        self.load_projections(projection_path)

        ownership_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['ownership_path']))
        self.load_ownership(ownership_path)

        player_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['player_path']))
        self.load_player_ids(player_path)

        boom_bust_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['boom_bust_path']))
        self.load_boom_bust(boom_bust_path)

        stats_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, self.config['stats_path']))
        self.load_stats(stats_path)
        
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


    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])
                    self.player_dict[player_name]['Boom'] = float(row['Boom%'])
                    self.player_dict[player_name]['Bust'] = float(row['Bust%'])

    # Load projected stats (minutes)
    def load_stats(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                if player_name in self.player_dict:
                    self.player_dict[player_name]['Minutes'] = int(row['Proj minutes'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')
                self.player_dict[player_name] = {'Fpts': 0, 'Position': None, 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0, 'Bust': 0}
                self.player_dict[player_name]['Fpts'] = float(row['Fpts'])
                self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))

                # Need to handle MPE on draftkings
                if self.site == 'dk':
                    self.player_dict[player_name]['Position'] = [pos for pos in row['Position'].split('/')]
                else:
                    self.player_dict[player_name]['Position'] = row['Position']

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
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        if self.use_randomness:
            self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev']) * lp_variables[player] for player in self.player_dict), 'Objective'
        else:
            self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict), 'Objective'

        # Set the salary constraints
        max_salary = 50000 if self.site == 'dk' else 60000
        self.problem += lpSum(self.player_dict[player]['Salary'] * lp_variables[player] for player in self.player_dict) <= max_salary

        if self.site == 'dk':
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
        else:
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
            try:
                self.problem.solve(PULP_CBC_CMD(msg=0))
            except PulpSolverError:
                print('Infeasibility reached - only generated {} lineups out of {}. Continuing with export.'.format(len(self.num_lineups), self.num_lineups))

            score = str(self.problem.objective)
            for v in self.problem.variables():
                score = score.replace(v.name, str(v.varValue))

            if i % 100 == 0:
                print(i)

            player_names = [v.name.replace('_', ' ') for v in self.problem.variables() if v.varValue != 0]
            fpts = eval(score)
            self.lineups[fpts] = player_names

            # Dont generate the same lineup twice
            if self.use_randomness:
                # Enforce this by lowering the objective i.e. producing sub-optimal results
                self.problem += lpSum(np.random.normal(self.player_dict[player]['Fpts'], self.player_dict[player]['StdDev'])* lp_variables[player] for player in self.player_dict)
            else:
                # Set a new random fpts projection within their distribution
                self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict) <= (fpts - 0.01)

            # Set number of unique players between lineups
            # data = sorted(self.player_dict.items())
            # for player_id, group_iterator in groupby(data):
            #     group = list(group_iterator)
            #     print(group)
            #     if len(group) == 1:
            #         continue
            #     variables = [variable for player, variable in group]
            #     solver.add_constraint(variables, None, SolverSign.LTE, 1)
            #     print(variables)
            # self.problem += len([ _id for _id in [self.player_dict[player]['ID'] * lp_variables[player] for player in self.player_dict] if _id not in set(player_names)]) >= self.num_uniques


    def output(self):
        unique = {}
        for fpts,lineup in self.lineups.items():
            if lineup not in unique.values():
                unique[fpts] = lineup

        self.lineups = unique
        self.format_lineups()
        num_uniq_lineups = OrderedDict(sorted(self.lineups.items(), reverse=False, key=lambda t: t[0]))
        self.lineups = {}
        for fpts,lineup in num_uniq_lineups.copy().items():
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
                  
        out_path = os.path.join(os.path.dirname(__file__), '../output/{}_optimal_lineups.csv'.format(self.site))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                
                f.write('PG,SG,SF,PF,C,G,F,UTIL,Fpts Proj,Fpts Sim,Salary,Own. Product,Minutes,Boom,Bust\n')
                for fpts, x in self.lineups.items():
                    salary = sum(self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
                    mins = sum(self.player_dict[player]['Minutes'] for player in x)
                    boom = sum(self.player_dict[player]['Boom'] for player in x)
                    bust = sum(self.player_dict[player]['Bust'] for player in x)
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{},{}'.format(
                        x[0].replace('#', '-'),self.player_dict[x[0]]['ID'],
                        x[1].replace('#', '-'),self.player_dict[x[1]]['ID'],
                        x[2].replace('#', '-'),self.player_dict[x[2]]['ID'],
                        x[3].replace('#', '-'),self.player_dict[x[3]]['ID'],
                        x[4].replace('#', '-'),self.player_dict[x[4]]['ID'],
                        x[5].replace('#', '-'),self.player_dict[x[5]]['ID'],
                        x[6].replace('#', '-'),self.player_dict[x[6]]['ID'],
                        x[7].replace('#', '-'),self.player_dict[x[7]]['ID'],
                        round(fpts_p, 2),round(fpts, 2),salary,own_p,mins,boom,bust
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write('PG,PG,SG,SG,SF,SF,PF,PF,C,Fpts Proj,Fpts Sim,Salary,Own. Product,Minutes,Boom,Bust\n')
                for fpts, x in self.lineups.items():
                    salary = sum(self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
                    mins = sum(self.player_dict[player]['Minutes'] for player in x)
                    boom = sum(self.player_dict[player]['Boom'] for player in x)
                    bust = sum(self.player_dict[player]['Bust'] for player in x)
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['ID'],x[0].replace('#', '-'),
                        self.player_dict[x[1]]['ID'],x[1].replace('#', '-'),
                        self.player_dict[x[2]]['ID'],x[2].replace('#', '-'),
                        self.player_dict[x[3]]['ID'],x[3].replace('#', '-'),
                        self.player_dict[x[4]]['ID'],x[4].replace('#', '-'),
                        self.player_dict[x[5]]['ID'],x[5].replace('#', '-'),
                        self.player_dict[x[6]]['ID'],x[6].replace('#', '-'),
                        self.player_dict[x[7]]['ID'],x[7].replace('#', '-'),
                        self.player_dict[x[8]]['ID'],x[8].replace('#', '-'),
                        round(fpts_p, 2),round(fpts, 2),salary,own_p,mins,boom,bust
                    )
                    f.write('%s\n' % lineup_str)

    def format_lineups(self):
        # TODO - fix dk
        if self.site == 'dk':
            return
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

        else:
            temp = self.lineups
            self.lineups = {}
            finalized = [None] * 9
            for fpts,lineup in temp.items():
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