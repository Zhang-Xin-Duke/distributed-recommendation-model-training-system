"""Reenter an existing managed training run; never replace roles on timeout."""
import argparse
import json
import pathlib
import sys

from managed_training_recovery import execute, PlacementRequired
from role_node import write


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True, type=pathlib.Path)
    a = p.parse_args(); value = json.loads(a.config.read_text())
    try:
        fleet=json.loads(pathlib.Path(value['fleet']).read_text())
        execute(fleet,value['run'],value['out'],fault=fleet.get('controller_test_fault'))
    except PlacementRequired:
        return 76
    except (ConnectionError, TimeoutError) as error:
        write(pathlib.Path(value['out']) / 'observation-pending.json', dict(error=str(error), live_handles_retained=True))
        return 75
    return 0


if __name__ == '__main__':
    sys.exit(main())
