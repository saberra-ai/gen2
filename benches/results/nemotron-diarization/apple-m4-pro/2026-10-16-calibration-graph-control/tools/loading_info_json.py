"""Preserve Transformers loading diagnostics while encoding its typed sets."""
import json


def jsonable_loading_info(value):
    def encode_set(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError('unsupported loading-info value: ' + type(item).__name__)
    return json.loads(json.dumps(value, default=encode_set))
