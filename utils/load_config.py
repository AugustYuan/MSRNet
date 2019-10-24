import os
import yaml


def load_config(dir):
    assert os.path.isfile(dir), ValueError('{} is not a file'.format(dir))
    assert dir.endswith('.yml'), ValueError('Expected a .yml file.')
    config = yaml.load(open(dir), Loader=yaml.FullLoader)
    return config