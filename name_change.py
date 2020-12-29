import fileinput

with fileinput.FileInput('projections.csv', inplace=True) as file:
    for line in file:
        print(line.replace('III', '')
        .replace('II','')
        .replace('IV', '')
        .replace('Jr.','')
        .replace('Sr.','')
        .replace('Moe Harkless', 'Maurice Harkless')
        .replace(' \"', '\"'), end='')

with fileinput.FileInput('ownership.csv', inplace=True) as file:
    for line in file:
        print(line.replace('III', '')
        .replace('II','')
        .replace('IV', '')
        .replace('Jr.','')
        .replace('Sr.','')
        .replace('Moe Harkless', 'Maurice Harkless')
        .replace(' \"', '\"'), end='')