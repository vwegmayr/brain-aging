"""
This modules allows to access the yaml config file
as if it was a python module.
"""
import importlib
import importlib.util
import os
import yaml


class ConfigParser:
    """ Wrapper class for the config.
    Before first use call parse_config_file
    or you will get an empty config object"""
    # Stores the parsed config

    config = {}

    @staticmethod
    def parse_config(filename):
        """ Read and parse yaml config file, initialized ConfigParser.config.
        Parses a yaml config file and returns a ConfigWrapper object
        with the attributes from the config file but with classes
        instead of strings as values.
        """
        filename = os.path.normpath(os.path.expanduser(filename))
        with open(filename) as config_file:
            config_dict = yaml.load(config_file)
        ConfigParser.import_python_classes(config_dict)
        ConfigParser.config = config_dict

        return config_dict

    @staticmethod
    def import_python_classes(obj):
        """ Replace 'module' and 'class' attributes with python objects """
        # Do the wrapper magic only if there is a 'module'
        # and a 'class' attribute(and obviously is dict)
        if isinstance(obj, dict):
            # If 'module' in obj and 'class' in obj:
            if key_contains("module", obj) and key_contains("class", obj):
                # Assign obj['module'] to the python module
                # instead of the string
                module_key = get_full_key("module", obj)
                class_key = get_full_key("class", obj)

                obj[module_key] = importlib.import_module(obj[module_key])
                # Assign obj['class'] to the python class instead of the string
                obj[class_key] = getattr(obj[module_key], obj[class_key])
                obj.pop(module_key)
            # Import module from any file which not necessarily needs to be in
            # a package
            if 'import' in obj and 'class' in obj:
                spec = importlib.util.spec_from_file_location(obj['class'],
                                                              obj['import'])
                obj['import'] = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(obj['import'])
                obj['class'] = getattr(obj['import'], obj['class'])
                if 'function' in obj:
                    obj['function'] = getattr(obj['class'], obj['function'])
            # Do the same thing for all other keys
            for key, value in obj.items():
                if key != 'module' and key != 'class' and "_conf" not in key:
                    ConfigParser.import_python_classes(value)

        # If the object is a list, continue the search for each item
        if isinstance(obj, list):
            for item in obj:
                ConfigParser.import_python_classes(item)


def key_contains(string, dict):
    for key in dict.keys():
        if string in key:
            return True


def get_full_key(string, dict):
    for key in dict.keys():
        if string in key:
            return key
