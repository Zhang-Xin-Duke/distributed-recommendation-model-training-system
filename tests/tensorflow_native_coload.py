"""Import the real TensorFlow and native clients in both process load orders."""
import argparse
import json
import pathlib
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=pathlib.Path)
    parser.add_argument('--gpu',action='store_true');args=parser.parse_args();records=[]
    for order in ('tensorflow-first','native-first'):
        imports='import tensorflow as tf; import _eps_native' if order=='tensorflow-first' else 'import _eps_native; import tensorflow as tf'
        code=imports+'; import json; assert _eps_native.read_lifecycle_cleanup; devices=tf.config.list_physical_devices("GPU"); '
        if args.gpu:code+='assert devices; '
        code+='print(json.dumps(dict(gpu_devices=[device.name for device in devices],native_loaded=True,tensorflow=tf.__version__)))'
        result=subprocess.run([sys.executable,'-X','faulthandler','-c',code],capture_output=True,text=True,timeout=45)
        if result.returncode:raise RuntimeError(order+' failed with exit '+str(result.returncode)+': '+result.stderr[-4000:])
        records.append(dict(order=order,exit_code=result.returncode,result=json.loads(result.stdout)))
    value=dict(test='tensorflow_native_coload',require_gpu=args.gpu,records=records)
    args.out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
