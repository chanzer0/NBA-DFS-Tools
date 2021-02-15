import json, csv, os, datetime, pytz, timedelta
import numpy as np
from pulp import *
from itertools import groupby


class NBA_Showdown_Optimizer:
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
    team_dict = {}

    def __init__(self, site=None, num_lineups=0, use_randomness=False, num_uniques=1):
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
        
    # Load config from file
    def load_config(self):
        with open(os.path.join(os.path.dirname(__file__), '../config.json')) as json_file: 
            self.config = json.load(json_file) 

    # Load player IDs for exporting
    def load_player_ids(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.site == 'dk':
                    player_name = row['Name'].replace('-', '#') + '_' + row['Roster Position']
                    if player_name in self.player_dict:
                        self.player_dict[player_name]['ID'] = int(row['ID'])
                        self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))
                else:
                    pos = ['_MVP', '_STAR', '_PRO', '_UTIL']
                    for p in pos:
                        player_name = row['Nickname'].replace('-', '#') + p
                        if player_name in self.player_dict:
                            self.player_dict[player_name]['ID'] = row['Id']
                            self.player_dict[player_name]['Salary'] = int(row['Salary'].replace(',',''))


    # Need standard deviations to perform randomness
    def load_boom_bust(self, path):
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.site == 'dk':
                    pos = ['_CPT', '_UTIL']
                    for p in pos:
                        player_name = row['Name'].replace('-', '#') + p
                        if player_name in self.player_dict:
                            self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])
                            self.player_dict[player_name]['Boom'] = float(row['Boom%'])
                            self.player_dict[player_name]['Bust'] = float(row['Bust%'])
                else:
                    pos = ['_MVP', '_STAR', '_PRO', '_UTIL']
                    for p in pos:
                        player_name = row['Name'].replace('-', '#') + p
                        if player_name in self.player_dict:
                            self.player_dict[player_name]['StdDev'] = float(row['Std Dev'])
                            self.player_dict[player_name]['Boom'] = float(row['Boom%'])
                            self.player_dict[player_name]['Bust'] = float(row['Bust%'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', '#')

                # Players can be rostered in any position slot, different constraints will apply
                if self.site == 'dk':
                    self.player_dict[player_name + '_CPT'] = {'Fpts': 0, 'Position': 'CPT', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_CPT']['Fpts'] = float(row['Fpts']) * 1.5
                    self.player_dict[player_name + '_CPT']['Salary'] = int(row['Salary'].replace(',','')) * 1.5
                    self.player_dict[player_name + '_CPT']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_CPT']['Team'] = row['Team']
                    self.player_dict[player_name + '_CPT']['Name'] = row['Name']
                    self.player_dict[player_name + '_UTIL'] = {'Fpts': 0, 'Position': 'UTIL', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_UTIL']['Fpts'] = float(row['Fpts'])
                    self.player_dict[player_name + '_UTIL']['Salary'] = int(row['Salary'].replace(',',''))
                    self.player_dict[player_name + '_UTIL']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_UTIL']['Team'] = row['Team']
                    self.player_dict[player_name + '_UTIL']['Name'] = row['Name']
                else:
                    self.player_dict[player_name + '_MVP'] = {'Fpts': 0, 'Position': 'MVP', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_MVP']['Fpts'] = float(row['Fpts']) * 2
                    self.player_dict[player_name + '_MVP']['Salary'] = int(row['Salary'].replace(',',''))
                    self.player_dict[player_name + '_MVP']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_MVP']['Team'] = row['Team']
                    self.player_dict[player_name + '_MVP']['Name'] = row['Name']
                    self.player_dict[player_name + '_STAR'] = {'Fpts': 0, 'Position': 'STAR', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_STAR']['Fpts'] = float(row['Fpts']) * 1.5
                    self.player_dict[player_name + '_STAR']['Salary'] = int(row['Salary'].replace(',',''))
                    self.player_dict[player_name + '_STAR']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_STAR']['Team'] = row['Team']
                    self.player_dict[player_name + '_STAR']['Name'] = row['Name']
                    self.player_dict[player_name + '_PRO'] = {'Fpts': 0, 'Position': 'PRO', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_PRO']['Fpts'] = float(row['Fpts']) * 1.2
                    self.player_dict[player_name + '_PRO']['Salary'] = int(row['Salary'].replace(',',''))
                    self.player_dict[player_name + '_PRO']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_PRO']['Team'] = row['Team']
                    self.player_dict[player_name + '_PRO']['Name'] = row['Name']
                    self.player_dict[player_name + '_UTIL'] = {'Fpts': 0, 'Position': 'UTIL', 'ID': 0, 'Salary': 0, 'StdDev': 0, 'Ownership': 0.1, 'Minutes': 0, 'Boom': 0.1, 'Bust': 0.1, 'Team': '', 'Name': ''}
                    self.player_dict[player_name + '_UTIL']['Fpts'] = float(row['Fpts'])
                    self.player_dict[player_name + '_UTIL']['Salary'] = int(row['Salary'].replace(',',''))
                    self.player_dict[player_name + '_UTIL']['Minutes'] = float(row['Minutes'])
                    self.player_dict[player_name + '_UTIL']['Team'] = row['Team']
                    self.player_dict[player_name + '_UTIL']['Name'] = row['Name']

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                if self.site == 'dk':
                    player_name = row['Name'].replace('-', '#')
                    if player_name + '_CPT' in self.player_dict:
                        self.player_dict[player_name + '_CPT']['Ownership'] = float(row['CPT %'])
                    if player_name + '_UTIL' in self.player_dict:
                        self.player_dict[player_name + '_UTIL']['Ownership'] = float(row['UTIL %'])
                else:
                    player_name = row['Name'].replace('-', '#')
                    if player_name + '_MVP' in self.player_dict:
                        self.player_dict[player_name + '_MVP']['Ownership'] = float(row['MVP %'])
                    if player_name + '_STAR' in self.player_dict:
                        self.player_dict[player_name + '_STAR']['Ownership'] = float(row['STAR %'])
                    if player_name + '_PRO' in self.player_dict:
                        self.player_dict[player_name + '_PRO']['Ownership'] = float(row['PRO %'])
                    if player_name + '_UTIL' in self.player_dict:
                        self.player_dict[player_name + '_UTIL']['Ownership'] = float(row['UTIL %'])
                
                

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
            # Need 1 captain and 5 utility players
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'CPT' == self.player_dict[player]['Position']) == 1
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'UTIL' == self.player_dict[player]['Position']) == 5
        else:
            # Need 1 MVP, 1 Star and 1 Pro and 2 FLEX players
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'MVP' == self.player_dict[player]['Position']) == 1
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'STAR' == self.player_dict[player]['Position']) == 1
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'PRO' == self.player_dict[player]['Position']) == 1
            self.problem += lpSum(lp_variables[player] for player in self.player_dict if 'UTIL' == self.player_dict[player]['Position']) == 2

        # no same CPTN and UTIL
        for p in self.player_dict.keys():
            self.problem += lpSum(lp_variables[player] for player in lp_variables if str(player).split('_')[0] == self.player_dict[p]['Name']) <= 1

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
        print('Lineups done generating. Outputting.')
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
                  
        out_path = os.path.join(os.path.dirname(__file__), '../output/{}_optimal_sd_lineups.csv'.format(self.site))
        with open(out_path, 'w') as f:
            if self.site == 'dk':
                f.write('CPT,UTIL,UTIL,UTIL,UTIL,UTIL,Salary,Fpts Proj,Own. Product,Minutes,Boom,Bust\n')
                for fpts, x in self.lineups.items():
                    salary = sum(self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
                    mins = sum(self.player_dict[player]['Minutes'] for player in x)
                    boom_p = np.prod([self.player_dict[player]['Boom']/100.0 for player in x])
                    bust_p = np.prod([self.player_dict[player]['Bust']/100.0 for player in x])
                    # optimal_p = np.prod([self.player_dict[player]['Optimal']/100.0 for player in x])
                    lineup_str = '{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{} ({}),{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['Name'],self.player_dict[x[0]]['ID'],
                        self.player_dict[x[1]]['Name'],self.player_dict[x[1]]['ID'],
                        self.player_dict[x[2]]['Name'],self.player_dict[x[2]]['ID'],
                        self.player_dict[x[3]]['Name'],self.player_dict[x[3]]['ID'],
                        self.player_dict[x[4]]['Name'],self.player_dict[x[4]]['ID'],
                        self.player_dict[x[5]]['Name'],self.player_dict[x[5]]['ID'],
                        salary,round(fpts_p, 2),own_p,mins,boom_p,bust_p
                    )
                    f.write('%s\n' % lineup_str)
            else:
                f.write('MVP,STAR,PRO,UTIL,UTIL,Salary,Fpts Proj,Own. Product,Minutes,Boom,Bust\n')
                for fpts, x in self.lineups.items():
                    salary = sum(self.player_dict[player]['Salary'] for player in x)
                    fpts_p = sum(self.player_dict[player]['Fpts'] for player in x)
                    own_p = np.prod([self.player_dict[player]['Ownership']/100.0 for player in x])
                    mins = sum(self.player_dict[player]['Minutes'] for player in x)
                    boom_p = np.prod([self.player_dict[player]['Boom']/100.0 for player in x])
                    bust_p = np.prod([self.player_dict[player]['Bust']/100.0 for player in x])
                    lineup_str = '{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}'.format(
                        self.player_dict[x[0]]['ID'],self.player_dict[x[0]]['Name'],
                        self.player_dict[x[1]]['ID'],self.player_dict[x[1]]['Name'],
                        self.player_dict[x[2]]['ID'],self.player_dict[x[2]]['Name'],
                        self.player_dict[x[3]]['ID'],self.player_dict[x[3]]['Name'],
                        self.player_dict[x[4]]['ID'],self.player_dict[x[4]]['Name'],
                        salary,round(fpts_p, 2),own_p,mins,boom_p,bust_p
                    )
                    f.write('%s\n' % lineup_str)
        print('Output done.')

    def format_lineups(self):
        if self.site == 'dk':
            temp = self.lineups.items()
            self.lineups = {}
            finalized = [None] * 6
            for fpts,lineup in temp:
                lineup = [p.rsplit(' ', 1)[0] + '_' + p.split(' ')[-1:][0] for p in lineup]
                for player in lineup:
                    if 'CPT' in self.player_dict[player]['Position']:
                        finalized[0] = player
                    elif 'UTIL' in self.player_dict[player]['Position']:
                        if finalized[1] is None:
                            finalized[1] = player
                        elif finalized[2] is None:
                            finalized[2] = player
                        elif finalized[3] is None:
                            finalized[3] = player
                        elif finalized[4] is None:
                            finalized[4] = player
                        elif finalized[5] is None:
                            finalized[5] = player
                        
                self.lineups[fpts] = finalized
                finalized = [None] * 6
        else:
            temp = self.lineups
            self.lineups = {}
            finalized = [None] * 5
            for fpts,lineup in temp.items():
                lineup = [p.rsplit(' ', 1)[0] + '_' + p.split(' ')[-1:][0] for p in lineup]
                for player in lineup:
                    if 'MVP' in self.player_dict[player]['Position']:
                        finalized[0] = player
                    elif 'STAR' in self.player_dict[player]['Position']:
                        finalized[1] = player
                    elif 'PRO' in self.player_dict[player]['Position']:
                        finalized[2] = player
                    elif 'UTIL' in self.player_dict[player]['Position']:
                        if finalized[3] is None:
                            finalized[3] = player
                        else:
                            finalized[4] = player
                self.lineups[fpts] = finalized
                finalized = [None] * 5