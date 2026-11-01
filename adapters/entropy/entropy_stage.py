"""Stage a committed parameter shard, optionally replacing this process with its server."""
import argparse
import json
import os
import pathlib
import sys
from checkpoint_restore import stage_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root',required=True)
    parser.add_argument('--checkpoint-id')
    parser.add_argument('--shard',required=True,type=int)
    parser.add_argument('--output-directory',required=True)
    parser.add_argument('--filesystem',choices=['local','tensorflow'],default='local')
    parser.add_argument('--exec-server')
    parser.add_argument('--service-config')
    parser.add_argument('--run')
    parser.add_argument('--listen')
    args=parser.parse_args()
    if args.exec_server and not args.service_config:parser.error('--exec-server requires --service-config')
    fs=None
    if args.filesystem=='tensorflow':
        import tensorflow as tf
        fs=tf.io.gfile
    report=stage_checkpoint(args.checkpoint_root,args.output_directory,args.shard,
                            filesystem=fs,checkpoint_id=args.checkpoint_id)
    print(json.dumps(report,sort_keys=True),file=sys.stderr if args.exec_server else sys.stdout,flush=True)
    if args.exec_server:
        server=str(pathlib.Path(args.exec_server).resolve())
        command=[server,'--config',args.service_config,'--restore',report['file']]
        if args.run:command+=['--run',args.run]
        if args.listen:command+=['--listen',args.listen]
        os.execv(server,command)


if __name__=='__main__':main()
