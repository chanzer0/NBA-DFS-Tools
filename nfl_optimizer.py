import json
import csv
from pulp import *

class NFL_Optimizer:
    problem = None
    config = None
    output_filepath = None
    num_lineups = None
    player_projections = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {}, 'DST': {}}
    player_salaries = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {}, 'DST': {}}
    player_ownership = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {}, 'DST': {}}
    roster_construction = {'QB': 1, 'RB': 2, 'WR': 3 + 1, 'TE': 1, 'DST': 1 }
    max_salary = 50000
    lineups = []

    def __init__(self):
        self.problem = LpProblem('NFL', LpMaximize)
        self.load_config()
        self.load_projections(self.config['projection_path'])
        self.load_ownership(self.config['ownership_path'])
        self.output_filepath = self.config['output_path']
        self.num_lineups = self.config['num_lineups']

    # Load config from file
    def load_config(self):
        with open('config.json') as json_file: 
            self.config = json.load(json_file) 

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                # projection
                self.player_projections[row['Position']][row['Name']] = float(row['Fpts'])
                # salary 
                self.player_salaries[row['Position']][row['Name']] = int(row['Salary'].replace(',',''))

    # Load ownership from file
    def load_ownership(self, path):
        # Read ownership into a dictionary
        with open(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.player_ownership[row['Position']][row['Name']] = row['Ownership %']

    def optimize(self):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot. 
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {pos: LpVariable.dict(pos, players, cat='Binary') for pos, players in self.player_projections.items()}

        projection_constraints = []
        salary_constraints = []
        position_constraints = []

        for position, players in lp_variables.items():
            # Set salary constraints
            salary_constraints += lpSum([self.player_salaries[position][player] * lp_variables[position][player] for player in players])
            # Set projections to maximize
            projection_constraints += lpSum([self.player_projections[position][player] * lp_variables[position][player] for player in players])
            # Set positional constraints
            self.problem += lpSum([lp_variables[position][player] for player in players]) == self.roster_construction[position]

        self.problem += lpSum(projection_constraints)
        self.problem += lpSum(salary_constraints) <= self.max_salary
        self.problem.solve()

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