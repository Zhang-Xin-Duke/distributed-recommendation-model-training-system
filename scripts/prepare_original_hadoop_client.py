#!/usr/bin/env python3
"""Prepare Apache Hadoop's real FS shell for original-framework local inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ARTIFACTS = [
    ('org/apache/hadoop', 'hadoop-client-api', '3.3.6'),
    ('org/apache/hadoop', 'hadoop-client-runtime', '3.3.6'),
    ('org/slf4j', 'slf4j-api', '1.7.36'),
    ('org/slf4j', 'slf4j-simple', '1.7.36'),
    ('commons-logging', 'commons-logging', '1.1.3'),
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def prepare(directory, java):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    java = str(Path(shutil.which(java) or java).resolve(strict=True))
    libraries = directory/'lib'
    libraries.mkdir(exist_ok=True)
    records = []
    for group, artifact, version in ARTIFACTS:
        name = f'{artifact}-{version}.jar'
        url = f'https://repo.maven.apache.org/maven2/{group}/{artifact}/{version}/{name}'
        expected = fetch(url+'.sha1').decode().split()[0].lower()
        if len(expected) != 40 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('invalid Maven checksum for '+name)
        path = libraries/name
        if not path.is_file() or hashlib.sha1(path.read_bytes()).hexdigest() != expected:
            content = fetch(url)
            if hashlib.sha1(content).hexdigest() != expected:
                raise ValueError('Maven artifact checksum differs: '+name)
            temporary = path.with_suffix('.partial')
            temporary.write_bytes(content)
            temporary.replace(path)
        records.append(dict(name=name, url=url, bytes=path.stat().st_size,
            sha1=expected, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    executable = directory/'bin/hadoop'
    executable.parent.mkdir(exist_ok=True)
    configuration = directory/'conf'
    configuration.mkdir(exist_ok=True)
    (configuration/'core-site.xml').write_text(
        '<?xml version="1.0"?>\n<configuration><property>'
        '<name>fs.defaultFS</name><value>file:///</value>'
        '</property></configuration>\n')
    # Launch the unmodified FsShell/VersionInfo classes; no filesystem commands
    # or listing results are reimplemented in the launcher.
    executable.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
java='''+repr(java)+'''
classpath=os.pathsep.join([str(root/'conf'),*[str(p) for p in sorted((root/'lib').glob('*.jar'))]])
if len(sys.argv)>1 and sys.argv[1]=='fs':
    command=[java,'-cp',classpath,'org.apache.hadoop.fs.FsShell',*sys.argv[2:]]
elif sys.argv[1:]==['version']:
    command=[java,'-cp',classpath,'org.apache.hadoop.util.VersionInfo']
else:
    raise SystemExit('this client package supplies hadoop fs and hadoop version')
os.execv(java,command)
''')
    executable.chmod(0o755)
    probe = directory/'probe'
    probe.mkdir(exist_ok=True)
    source = probe/'sample.bin'
    content = b'original-framework-local-filesystem\x00\xff\n'
    source.write_bytes(content)
    destination = probe/'copied.bin'
    if destination.exists():
        destination.unlink()
    version = subprocess.run([str(executable), 'version'], check=True, capture_output=True, text=True)
    assert version.stdout.splitlines()[0] == 'Hadoop 3.3.6'
    listing = subprocess.run([str(executable), 'fs', '-ls', '-d', str(source)],
        check=True, capture_output=True, text=True)
    fields = listing.stdout.strip().split()
    assert len(fields) == 8 and int(fields[4]) == len(content) and fields[7] == str(source)
    actual = subprocess.run([str(executable), 'fs', '-cat', str(source)], check=True, capture_output=True)
    assert actual.stdout == content
    subprocess.run([str(executable), 'fs', '-cp', str(source), str(destination)],
                   check=True, capture_output=True)
    assert destination.read_bytes() == content
    subprocess.run([str(executable), 'fs', '-test', '-e', str(destination)],
                   check=True, capture_output=True)
    subprocess.run([str(executable), 'fs', '-rm', '-skipTrash', str(destination)],
                   check=True, capture_output=True)
    assert not destination.exists()
    report = dict(hadoop_version='3.3.6', java=java,
        java_version=subprocess.run([java, '-version'], check=True, capture_output=True,
                                    text=True).stderr.splitlines()[0],
        executable=str(executable), artifacts=records,
        checks=['real FsShell ls metadata', 'binary cat', 'local copy', 'existence', 'remove'],
        listing=listing.stdout.strip(), distributed_hdfs_verified=False)
    (directory/'client-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--java', default='java')
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.java)))


if __name__ == '__main__':
    main()
