"""Compile the native operation module with the installed TensorFlow wheel ABI."""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tensorflow as tf

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=pathlib.Path,required=True)
args=parser.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
args.out.parent.mkdir(parents=True,exist_ok=True)
command=[os.environ.get('CXX','c++'),'-shared','-fPIC','-O2',str(root/'adapters/entropy/tensorflow_ops.cc'),
         '-I'+str(root/'include'),*tf.sysconfig.get_compile_flags(),*tf.sysconfig.get_link_flags(),
         '-o',str(args.out)]
if sys.platform=='darwin':
    # TensorFlow 2.20 SDK specializes std::is_signed for quantized types; newer
    # Apple libc++ diagnoses those unused SDK declarations as an error.
    command+=['-Wno-invalid-specialization','-Wl,-rpath,'+tf.sysconfig.get_lib()]
else:command+=['-ldl','-pthread','-Wl,-rpath,'+tf.sysconfig.get_lib()]
subprocess.run(command,check=True)
args.out.with_suffix('.build.json').write_text(json.dumps({'tensorflow':tf.__version__,'command':command,
    'compile_flags':tf.sysconfig.get_compile_flags(),'link_flags':tf.sysconfig.get_link_flags()},indent=2)+'\n')
print('Built TensorFlow native CPU/GPU operations:',args.out)
