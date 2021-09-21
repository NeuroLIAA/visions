import json

def load_dict_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)

def save_to_json(file, data):
    with open(file, 'w') as json_file:
        json.dump(data, json_file, indent=4)