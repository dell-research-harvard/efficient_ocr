from glob import glob
import os
import copy

def dictmerge(a: dict, b: dict, path=None) -> dict:
    """Merge two dictionaries recursively without mutating the inputs.
    
    This version uses deepcopy so that even nested objects are completely independent.
    Returns a new dictionary that is the result of merging b into a.
    """
    if path is None:
        path = []
    # Make a deep copy of a to avoid mutating the original dictionary.
    result = copy.deepcopy(a)
    
    for key in b:
        if key in result:
            if isinstance(result[key], dict) and isinstance(b[key], dict):
                # Recursively merge the nested dictionaries.
                result[key] = dictmerge(result[key], b[key], path + [str(key)])
            elif result[key] != b[key]:
                # If values differ, copy b's value deeply.
                result[key] = copy.deepcopy(b[key])
        else:
            result[key] = copy.deepcopy(b[key])
    return result


def get_path(d, contains="", ends_in="", ext="", priority="newest"):

    contains = '*' + contains if len(contains) > 0 else ""
    ext = '.' + ext if len(ext) > 0 else ""
    paths = glob(os.path.join(d, f"**/{contains}*{ends_in}{ext}"), recursive=True)

    if len(paths) == 0:
        Warning("No paths found")
        return None

    if priority == "newest":
        path = max(paths, key=os.path.getctime)
    elif priority == "oldest":
        path = min(paths, key=os.path.getctime)
    else:
        raise NotImplementedError
    
    return path


def dir_is_empty(d):
    return len(os.listdir(d)) == 0


def last_in_path(p, sep="/"):
    return p.split(sep)[-1]


def all_but_last_in_path(p, sep="/"):
    return sep.join(p.split(sep)[:-1])

