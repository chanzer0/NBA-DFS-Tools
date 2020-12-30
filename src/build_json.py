import csv, json, sys, os


def main(arguments):
    if len(arguments) != 2:
        print('`python .\build_json.py <site>`')
        exit()

    site = arguments[1]
    player_dict = {}
    # Read projections into a dictionary
    projection_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, 'projections.csv'))
    with open(projection_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            player_dict[row['Name']] = {'Fpts': 'N/A', 'Ownership %': '0', 'Boom %': '0', 'Optimal %': '0'}
            player_dict[row['Name']]['Fpts'] = row['Fpts']

    # Read ownership into dictionary       
    ownership_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, 'ownership.csv'))
    with open(ownership_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'] in player_dict:
                player_dict[row['Name']]['Ownership %'] = row['Ownership %']

    # Read boom/bust into dictionary
    boom_bust_path = os.path.join(os.path.dirname(__file__), '../{}_data/{}'.format(site, 'boom_bust.csv'))
    with open(boom_bust_path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'] in player_dict:
                player_dict[row['Name']]['Boom %'] = row['Boom%']
                player_dict[row['Name']]['Optimal %'] = row['Optimal%']


    # Store results
    result_path = os.path.join(os.path.dirname(__file__), '../output/playerData.json')
    with open(result_path, 'w') as fp:
        json.dump(player_dict, fp)


if __name__ == "__main__":
    main(sys.argv)