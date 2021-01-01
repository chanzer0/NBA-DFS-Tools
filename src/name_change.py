import fileinput, os


projection_path = os.path.join(os.path.dirname(__file__), '../fd_data/projections.csv')
with fileinput.FileInput(projection_path, inplace=True) as file:
    for line in file:
        print(line.replace('III', '')
        .replace('II','')
        .replace('IV', '')
        .replace('Jr.','')
        .replace('Sr.','')
        .replace('Moe Harkless', 'Maurice Harkless')
        .replace('PJ Washington', 'P.J. Washington')
        .replace('Jakarr Sampson', 'JaKarr Sampson')
        .replace(' \"', '\"'), end='')

ownership_path = os.path.join(os.path.dirname(__file__), '../fd_data/ownership.csv')
with fileinput.FileInput(ownership_path, inplace=True) as file:
    for line in file:
        print(line.replace('III', '')
        .replace('II','')
        .replace('IV', '')
        .replace('Jr.','')
        .replace('Sr.','')
        .replace('Moe Harkless', 'Maurice Harkless')
        .replace('PJ Washington', 'P.J. Washington')
        .replace('Jakarr Sampson', 'JaKarr Sampson')
        .replace(' \"', '\"'), end='')