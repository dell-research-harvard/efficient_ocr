from glob import glob
import os


def dictmerge(a: dict, b: dict, path=[]):
    """h/t: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries"""
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                dictmerge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key] # raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


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

