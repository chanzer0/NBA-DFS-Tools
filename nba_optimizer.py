import json
import csv
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
                player_name = row['Name'].replace('-', ' ')
                self.player_dict[player_name]['ID'] = int(row['ID'])

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                player_name = row['Name'].replace('-', ' ')
                self.player_dict[player_name] = {'Fpts': None, 'Position': [], 'ID': 0, 'Salary': None, 'Ownership': None}
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
                player_name = row['Name'].replace('-', ' ')
                self.player_dict[player_name]['Ownership'] = float(row['Ownership %'])

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot. 
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        # lp_variables = {'PG': 'LeBron James' : 'PG_LeBron James', 'Kyrie Irving': 'PG_Kyrie_Irving', .... , 'SG' : .... , etc... }
        lp_variables = {player: LpVariable(player, cat='Binary') for player, _ in self.player_dict.items()}

        # set the objective - maximize fpts
        self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict)

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

            # id_sum = sum(self.player_dict[v.name.replace('_', ' ')]['ID'] for v in self.problem.variables() if v.varValue != 0)
            player_names = [v.name.replace('_', ' ') for v in self.problem.variables() if v.varValue != 0]
            fpts_total = round(sum(self.player_dict[v.name.replace('_', ' ')]['Fpts'] for v in self.problem.variables() if v.varValue != 0), 2)
            print('' + str(fpts_total) + ' - ' + str(player_names))
            self.lineups[fpts_total] = player_names

            # # Dont generate the same lineup twice - enforce this through the sum of player Ids!
            # self.problem += lpSum(self.player_dict[player]['ID'] * lp_variables[player] for player in self.player_dict) != id_sum
           
            self.problem += lpSum(self.player_dict[player]['Fpts'] * lp_variables[player] for player in self.player_dict) <= (fpts_total - 0.01)
            # self.output()

        print(self.lineups)

    def output(self):
        div = '---------------------------------------\n'
        print('Variables:\n')
        score = str(self.problem.objective)
        constraints = [str(const) for const in self.problem.constraints.values()]
        for v in self.problem.variables():
            score = score.replace(v.name, str(v.varValue))
            constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
            if v.varValue != 0:
                print(v.name, '=', v.varValue)
        print(div)
        print('Constraints:')
        for constraint in constraints:
            constraint_pretty = ' + '.join(re.findall('[0-9\.]*\*1.0', constraint))
            if constraint_pretty != '':
                print('{} = {}'.format(constraint_pretty, eval(constraint_pretty)))
        print(div)
        print('Score:')
        score_pretty = ' + '.join(re.findall('[0-9\.]+\*1.0', score))
        print('{} = {}'.format(score_pretty, eval(score)))

        print(self.lineups)
        # with open(self.output_filepath, 'w') as f:
        #     f.write('QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts,Salary,Ownership\n')
        #     for x in final:
        #         fpts = sum(float(projection_dict[player]['Fpts']) for player in x)
        #         salary = sum(int(projection_dict[player]['Salary'].replace(',','')) for player in x)
        #         own = sum(float(ownership_dict[player]) for player in x)
        #         lineup_str = '{},{},{},{},{},{},{},{},{},{},{},{}'.format(
        #             x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],fpts,salary,own
        #         )
        #         f.write('%s\n' % lineup_str)