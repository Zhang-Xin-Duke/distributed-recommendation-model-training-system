"""Install external tables before executing the unchanged training entry point."""
import argparse
import importlib
import json
import runpy
import sys

from entropy_install import JobConfiguration,install


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameter-config',required=True)
    parser.add_argument('--train-module',default='tensorswitch.train')
    entry=parser.add_mutually_exclusive_group(required=True)
    entry.add_argument('--entry-module')
    entry.add_argument('--entry-script')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    with open(args.parameter_config,encoding='utf-8') as stream:
        configuration=JobConfiguration(json.load(stream))
    train=importlib.import_module(args.train_module)
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    previous=sys.argv
    sys.argv=[args.entry_module or args.entry_script,*arguments]
    try:
        with install(train,configuration):
            if args.entry_module:
                runpy.run_module(args.entry_module,run_name='__main__',alter_sys=True)
            else:
                runpy.run_path(args.entry_script,run_name='__main__')
    finally:
        sys.argv=previous


if __name__=='__main__':main()
