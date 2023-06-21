import yaml
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module


class Factory:
    def __init__(self):
        self._creators = {}

    def register(self, task_name, creator):
        if task_name in self._creators:
            raise ValueError(f"{task_name} task already exists!")
        self._creators[task_name] = creator

    def register_all_subclasses(self, taskClass):
        for subclass in self.get_subclasses(taskClass):
            self.register(subclass.__name__, subclass)

    def get_class(self, task_name):
        creator = self._creators.get(task_name)
        if not creator:
            raise ValueError(task_name)
        return creator

    def get_subclasses(self, cls):
        for subclass in cls.__subclasses__():
            yield from self.get_subclasses(subclass)
            yield subclass


class ObjectDict(dict):
    def __init__(self, dict_input):
        for k, v in dict_input.items():
            if isinstance(v, dict):
                v = ObjectDict(v)
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.

    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: ObjectDict.
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    if kwargs:
        for name, value in kwargs.items():
            config[name] = value

    return ObjectDict(config)
