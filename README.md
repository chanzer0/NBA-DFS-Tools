# NBA Optimizer and GPP Utilities

Packaged in this repository is an NBA optimzier for DraftKings and FanDuel, along with other tools you might find useful in helping you win your Cash games, Head-to-heads and Tournaments. You'll find installation and usage instructions below. Data input is in the format of [Awesemo](https://www.awesemo.com/join/#/) CSV export. Awesemo has great player projection and ownership predictions that can't be matched. Sign up and join in premium slack where DFS pros, including Alex "Awesemo" Baker himself, can answer your quetsions and help you win.

## Installation

### System requirements

- To run the tools, you will need to [install python](https://www.python.org/downloads/) if you don't already have it. These tools were bult on Python 3.8.2, and may be incompatible with outdated versions of Python.
- In addition to the base python installation, you will need the following packages:
  - [PuLP](https://pypi.org/project/PuLP/) - `pip install pulp`. This is the linear programming solver - the "optimizer" if you will.
  - [timedelta](https://pypi.org/project/timedelta/) - `pip install timedelta`. This package makes it easy to interpret dates for late swaptimizing lineups.
  - [pytz](https://pypi.org/project/pytz/) - `pip install pytz`. Another helpful package for interpreting dates and late swaptimizing
  - [numpy](https://pypi.org/project/numpy/) - `pip install numpy`. This package makes data manipulation and handling matrices easier.

To install these tools, you may either clone this repository or download the repository as a ZIP file (see image below) and extract it to the directory of your choosing.

![Download image](readme_images/download.png)

After you have cloned or downloaded the code base, you must import player contest data from DraftKings or FanDuel. Reference the screenshots below for your relative site. You will need to rename these files to `player_ids.csv`, and place into their relative directory (`dk_data/` or `fd_data/`). These directories should be folders located in the same directory as `src/` and `output/`, and will hold relevant data for each site.

After you have the player data, you must import data from Awesemo, namely the projections, ownership, boom/bust tool. Download them as CSV, and rename them to match the image below. These will go in either `dk_data/` or `fd_data/` depending on which data you downloaded.

![Directory image](readme_images/directory.png)

## Usage

To use the tools, you will need to open a windows console or powershell terminal in the same directory as this repository. To do this, go to the root directory and then navigate to `File > Open Windows Powershell` as seen below.

![Shell image](readme_images/shell.png)

To run the tools, the generic usage template is as follows:
`python .\main.py <site> <process> <num_lineups> <num_uniques> <use_rand>`

Where:
`<site>` is:

- `dk` for DraftKings. Note for DraftKings, players are output in alphabetical order and must be re-ordered into their positions before uploading back to DK.
- `fd` for FanDuel. Note for FanDuel, you must also run `python .\name_change.py` before any crunching, as some player names differ between Awesemo projections and FanDuel's player data

`<process>` is:

- `opto` for running optimal lineup crunches, with or without randomness
- `sim` for running GPP simulations

  - Usage #1 allows you to run arbitrary simulations for any field size and number of iterations, without regards to a real contest structure. The usage for this is: `python .\main.py <site> sim <field_size> <num_iterations>`, where `<field_size>` is the known entrant size, and `<num_iterations>` is the number of times you wish to simulate the tournament.
  - Usage #2 allows you to specify an actual DraftKings contest, which will dictate `<field_size>`. You will specify the number of iterations, but specifying the contest allows the simulation to take ROI into account, since the payout structure and entry fee is known. The usage for this is: `python .\main.py <site> sim cid <num_iterations>`. To execute this usage, you will need a `contest_structure.csv` file in the structure of the image shown below. You can obtain this fairly quickly by opening up the contest details overlay and copy/pasting the prize payouts section into Excel or Google sheets, then using `Ctrl+H` to get rid of 'st', 'nd', 'rd', 'th', etc...

        ![Contest structure input](readme_images/contest_structure_input.png)

    - Additionally, you may opt to upload lineups from a file rather than have them randomly generated/simulated. To specify this option, you will add `file` as a flag in your command like so: `python .\main.py <site> sim cid file 10000`. You must have an input file called `tournament_lineups.csv` in the base input directory. This allows you to upload specifically-tailored lineups that you feel are more representative of your contest than the ones generated. It also has the added benefit of being much faster than generating lineups. For example, you may take the output of the `opto` process, and rename the file to `tournament_lineups.csv`, and use those as your input for the `sim` process.

- `sd` for running showdown crunches, with or without randomness

`<num_lineups>` is the number of lineups you want to generate when using the `opto` process.

`<num_uniques>` defines the number of players that must differ from one lineup to the next. **NOTE** - this is enforced _after_ crunching, so 1000 lineups may be whittled down to 7-800 depending on this constraint. Enforcing this _after_ crunching aids in the speed of crunching. Expect approximately 5-15% of lineups to be "culled" by this constraint and plan accordingly. The more players available to crunch, the lower this percentage will be.

`<use_randomess>` is:

- `rand` if you want to incorporate [Standard Normal](https://www.mathsisfun.com/data/standard-normal-distribution.html) randomness
- `none` (or any string that isn't 'rand') if you want to crunch strict optimals with no randomness added.

For example, to generate 1000 lineups for DraftKings, with 3 uniques and randomness, I would execute the following:
`python .\main.py dk opto 1000 3 rand`

The image below shows what the shell/terminal should look like when executing this. You may safely ignore the PuLP overwriting warning, as we must overwrite the linear programming objective with the updated random projections.

![Example usage](readme_images/usage.png)

## Config

In the base directory, you will find `config.json`, which has a few template options for you to limit players from teams, and make groups of players you want a limit on. The structure for them is as follows:

```
"at_least": {
    "2": [
        ["Stephen Curry", "Domantas Sabonis", "Joel Embiid"], // This will use at least 2 of these players
    ],
    "1": [
        ["LeBron James", "Jayson Tatum"], // This will use at least 1 of these players
    ],
},
"at_most": {
    "1": [
        ["Clint Capela", "Onyeka Okongwu"], // Will use at most 1 of these players, good for players and their direct backups
        ["Evan Mobley", "Kevin Love"],
    ],
    "2": [
        ["Jevon Carter", "Grayson Allen", "Wesley Matthews"], // Will use at most 2 of these players, good for players that may close or sit bench
    ],
},
"team_limits": {
    "MIL": "3", // Will use at most 3 players from Milwaukee
},
"global_team_limit": "5" // This will limit all teams to a maximum of 5 players
```

## Output

Data is stored in the `output/` directory. Note that subsequent runs of the tool will overwrite previous output files, so either move them or rename them if you wish to preseve them. From there, you may upload these `.csv` files into Excel, and "pretty them up" - this can be seen below

### `opto` Process

![Example output](readme_images/opto_output.png)

### `sim` Process

![Example output](readme_images/sim_output.png)
