

# h/t: https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def dictmerge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                dictmerge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key] # raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def dictlistmerge(d, l, v):
    top = l.pop(0)
    if top in d:
        if isinstance(d[top], dict):
            dictlistmerge(d[top], l, v)
        else:
            d[top] = v
    else:
        raise KeyError(f"Key [{top}] not in dictionary {d}")
    return d

