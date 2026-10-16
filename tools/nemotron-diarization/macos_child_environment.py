"""Prepare an offline sandbox child and its launcher environment."""


def sandboxed_command(arguments, profile, environment):
    parent = dict(environment)
    child = list(map(str, arguments))
    trace = parent.pop('DYLD_PRINT_LIBRARIES', None)
    if trace is not None:
        child = ['/usr/bin/env', 'DYLD_PRINT_LIBRARIES=' + trace, *child]
    return ['/usr/bin/sandbox-exec', '-f', str(profile), *child], parent
