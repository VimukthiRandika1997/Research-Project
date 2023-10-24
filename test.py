import yaml

def read_yaml_file(path_to_file):
    with open(path_to_file, 'r') as f:
        data = yaml.safe_load(f)

    return data


if __name__ == '__main__':
    data = read_yaml_file('./test.yaml')
    print(data)
    print('EINv1' in data.keys())