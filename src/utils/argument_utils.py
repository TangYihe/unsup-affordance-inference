"""load and save YAML config file. Originally from VoxPoser"""
import os
import sys
import yaml
import json

class ConfigDict(dict):
    def __init__(self, config):
        """recursively build config"""
        # self.config = config
        for key, value in config.items():
            if isinstance(value, str) and value.lower() == 'none':
                value = None
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            else:
                self[key] = value

    def __getattr__(self, key):
        if key in self:
            return self[key]
        elif key == 'config':
            return self
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]
    def __getstate__(self):
        return self.config
    def __setstate__(self, state):
        self.config = state
        self.__init__(state)

    def update(self, config):
        """update with another dict"""
        if isinstance(config, ConfigDict):
            config = config.convert_to_dict()
        for key, value in config.items():
            if isinstance(value, dict):
                self[key].update(value)
            else:
                self[key] = value

    def convert_to_dict(self):
        """convert to dict"""
        config = {}
        for key, value in self.items():
            if isinstance(value, ConfigDict):
                config[key] = value.convert_to_dict()
            else:
                config[key] = value
        return config

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_yaml_config(config_path=None):
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_yaml_config(config_path)
    config = ConfigDict(config)
    return config

def eval_str_to_lst(query_str):
    """
    Parse a string in format [a, b, c] to a list
    """
    query_str = query_str.replace('[', '').replace(']', '')
    query_lst = query_str.split(',')
    query_lst = [q.strip() for q in query_lst]
    return query_lst

def get_command_line_args(argv, to_config_dict=True):
    """
    Utility function to parse all command line arguments and return them as a dictionary.
    If argument is in format 'key1.key2=value', it will be parsed as a nested dictionary.
    """
    args_dict = {}
    for arg in argv[1:]:  # Skip the first argument (script name)
        if '=' in arg:
            key, value = arg.split('=', 1)
            key = key.lstrip('-')  # Remove leading dashes

            # Try to convert value to int, float, bool, or leave as string
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass

            if isinstance(value, str) and '[' in value and ']' in value:
                value = eval_str_to_lst(value)
            
            # Check for hierarchy in the key (indicated by '.')
            if '.' in key:
                sub_keys = key.split('.')
                current_dict = args_dict
                # Iterate through sub_keys to create nested dictionaries
                for sub_key in sub_keys[:-1]:
                    if sub_key not in current_dict:
                        current_dict[sub_key] = {}
                    current_dict = current_dict[sub_key]
                current_dict[sub_keys[-1]] = value
            else:
                args_dict[key] = value

    if to_config_dict:
        args_dict = ConfigDict(args_dict)
    return args_dict

def save_config(config, config_path):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    if type(config) != dict:
        print("Converting config to dict")
        config = config.convert_to_dict()
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

# def main():
#     config = get_yaml_config(config_path='./configs/sim_env/empty_scene_fetch.yaml')
#     from IPython import embed; embed()

if __name__ == '__main__':
    from IPython import embed; embed(); exit(0)
    # main()