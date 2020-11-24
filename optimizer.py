import json
import csv
from pulp import *

def convert_to_lp(name):
    if name[0].isdigit():
        name = 'x' + name
    return name.replace("-","(hyphen)").replace("+","(plus)").replace("[","(leftBracket)").replace("]","(rightBracket)").replace(" ","(space)").replace(">","(greaterThan)").replace("/","(slash)")

def convert_to_str(name):
    if name[1].isdigit():
        name = name[1:]
    return name.replace("(hyphen)","-").replace("(plus)","+").replace("(leftBracket)","[]").replace("(rightBracket)","]").replace("(space)"," ").replace("(greaterThan)",">").replace("(slash)","/")

def summary(prob):
    div = '---------------------------------------\n'
    print("Variables:\n")
    score = str(prob.objective)
    constraints = [str(const) for const in prob.constraints.values()]
    for v in prob.variables():
        score = score.replace(v.name, str(v.varValue))
        constraints = [const.replace(v.name, str(v.varValue)) for const in constraints]
        if v.varValue != 0:
            print(v.name, "=", v.varValue)
    print(div)
    print("Constraints:")
    for constraint in constraints:
        constraint_pretty = " + ".join(re.findall("[0-9\.]*\*1.0", constraint))
        if constraint_pretty != "":
            print("{} = {}".format(constraint_pretty, eval(constraint_pretty)))
    print(div)
    print("Score:")
    score_pretty = " + ".join(re.findall("[0-9\.]+\*1.0", score))
    print("{} = {}".format(score_pretty, eval(score)))

DK_ROSTER = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "DST": 1,
    "FLEX": 1
}
MAX_SALARY = 50000

# Load our config
with open('config.json') as json_file: 
    config = json.load(json_file) 

PROJECTIONS_FILE = config['projection_path']
OWNERSHIP_FILE = config['ownership_path'] 
OUTPUT_FILE = config['output_path']
NUM_LINEUPS = config['num_lineups']

player_projections = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {} , 'DST': {}, 'FLEX': {}}
player_salaries = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {} , 'DST': {}, 'FLEX': {}}
player_ownership = {'QB': {}, 'RB': {}, 'WR': {}, 'TE': {} , 'DST': {}, 'FLEX': {}}
player_positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [] , 'DST': [], 'FLEX': []}

# Read projections into a dictionary
with open(PROJECTIONS_FILE) as file:
    reader = csv.DictReader(file)
    for row in reader:
        # projection
        player_projections[row['Position']][row['Name']] = float(row['Fpts'])
        # salary 
        player_salaries[row['Position']][row['Name']] = int(row['Salary'].replace(',',''))
        # position
        player_positions[row['Position']].append(row['Name'])

        # do the same if they're a flex player
        if row['Position'] == 'RB' or row['Position'] == 'WR' or row['Position'] == 'TE':
            player_projections['FLEX'][row['Name']] = float(row['Fpts'])
            player_salaries['FLEX'][row['Name']] = int(row['Salary'].replace(',',''))
            player_positions['FLEX'].append(row['Name'])

# Read ownership into a dictionary
with open(OWNERSHIP_FILE) as file:
    reader = csv.DictReader(file)
    for row in reader:
        player_ownership[row['Position']][row['Name']] = row['Ownership %']

# Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
# We will use PuLP as our solver - https://coin-or.github.io/pulp/

# We want to create a variable for each roster slot. 
# There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
lp_variables = {k: LpVariable.dict(k, v, cat="Binary") for k, v in player_projections.items()}

problem = LpProblem("NFL", LpMaximize)
projection_constraints = []
salary_constraints = []
position_constraints = []

for k, v in lp_variables.items():
    # Set salary constraints
    salary_constraints += lpSum([player_salaries[k][i] * lp_variables[k][i] for i in v])
    # Set projections to maximize
    projection_constraints += lpSum([player_projections[k][i] * lp_variables[k][i] for i in v])
    # Set positional constraints
    problem += lpSum([lp_variables[k][i] for i in v]) <= DK_ROSTER[k]
    # Set flex position constraints
    # problem += lpSum([FLEX_ELIGIBILITY[k] * lp_variables[k][i] for i in v]) <= 5
    
problem += lpSum(projection_constraints)
problem += lpSum(salary_constraints) <= MAX_SALARY
problem.solve()
summary(problem)



# with open(OUTPUT_FILE, 'w') as f:
#     f.write("QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts,Salary,Ownership\n")
#     for x in final:
#         fpts = sum(float(projection_dict[player]['Fpts']) for player in x)
#         salary = sum(int(projection_dict[player]['Salary'].replace(',','')) for player in x)
#         own = sum(float(ownership_dict[player]) for player in x)
#         lineup_str = "{},{},{},{},{},{},{},{},{},{},{},{}".format(
#             x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],fpts,salary,own
#         )
#         f.write("%s\n" % lineup_str)