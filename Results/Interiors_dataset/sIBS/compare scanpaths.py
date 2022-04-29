import json
from os import path
def load_dict_from_json(json_file_path):
    if not path.exists(json_file_path):
        return {}
    else:
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)


scanpaths_python= load_dict_from_json('Scanpaths_python.json')
scanpaths_matlab= load_dict_from_json('Scanpaths_matlab.json')
scanpaths_diff = {}
amount = 0
for x in scanpaths_matlab.keys():
    matlab_scanpath=scanpaths_matlab[x]
    python_scanpath= scanpaths_python[x]
    if matlab_scanpath['X'] != python_scanpath['X'] or matlab_scanpath['Y'] != python_scanpath['Y']:
        amount += 1
        scanpaths_diff[x] = {'matlab_X' : matlab_scanpath['X'], 'matlab_Y': matlab_scanpath['Y'],'python_X': python_scanpath['X'], 'python_Y':python_scanpath['Y']}
print(amount)
with open('scanpaths_diff.json', 'w') as json_file:
    json.dump(scanpaths_diff, json_file, indent=4)


