import json
import csv
import numpy
import pulp
import random

PROJECTIONS_FILE = ''
OWNERSHIP_FILE = '' 
OUTPUT_FILE = ''

player_projections = {}
player_salaries = {}
player_ownership = {}
player_positions = {'QB': [], 'RB': [], 'WR': [], 'TE': [] , 'DST': []}

# Read projections into a dictionary
with open(PROJECTIONS_FILE) as file:
    reader = csv.DictReader(file)
    for row in reader:
        # projection
        player_projections[row['Name']] = row

        # salary 
        player_salaries[row['Name']] = int(row['Salary'].replace(',',''))

        # position
        player_positions[row['Position']].append(row['Name'])

# Read ownership into a dictionary
with open(OWNERSHIP_FILE) as file:
    reader = csv.DictReader(file)
    for row in reader:
        ownership_dict[row['Name']] = row['Ownership %']

# If no ownership is included for projected players, set their ownership to 0
for key,v in projection_dict.items():
    if key not in ownership_dict:
        print ('Player \'' + str(key) + '\' has no ownership - settings to 0.00 manually')
        ownership_dict[key] = '0.00'

print(player_projections)
print(player_salaries)
print(player_ownership)
print(player_positions)
