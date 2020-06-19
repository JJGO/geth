import yaml
import hashlib
from collections.abc import Mapping, MutableMapping


def dict_recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = dict_recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def expand_dots(d):
    # expand_dots({"a.b.c": 1, "J":2, "a.d":2, "a.b.d":3})
    newd = {}
    for k, v in d.items():
        if '.' in k:
            pre, post = k.split('.', maxsplit=1)
            u = expand_dots({post: v})
            if pre in newd:
                newd[pre] = dict_recursive_update(newd[pre], u)
            else:
                newd[pre] = u
        else:
            newd[k] = v
    return newd


def expand_keys(d):
    expanded = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            for k2, v2 in expand_keys(v).items():
                expanded[f"{k}.{k2}"] = v2
        else:
            expanded[k] = v
    return expanded


class Config(MutableMapping):

    def __init__(self, **kwargs):
        self.cfg = kwargs

    def __getitem__(self, key):
        if '.' in key:
        else:

    def __delitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cfg})"

    def __str__(self):
        return yaml.dump(self.cfg, indent=2)

    @staticmethod
    def load(file):
        with open(file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return Config(cfg)

    def flatten(self):
        return expand_keys(self.cfg)

    def dump(self, file):
        with open(file, 'w') as f:
            yaml.dump(self.cfg, f, indent=2)

    def digest(self, ignore=None):
        cfg = allbut(self.cfg, ignore)
        cfg_dump = yaml.dump(cfg, sort_keys=True).encode('uft-8')
        return hashlib.md5(cfg_dump).hexdigest()


