#!/usr/bin/env python3
"""Generate the original training/control Python protocols in a private directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace, output, python):
    workspace = workspace.resolve(strict=True)
    entropy = workspace / "entropy"
    api = workspace / "repositories/api/protos"
    sources = sorted((entropy / "proto").glob("*.proto"))
    schema = api / "galaxy/kfc_data/embedding_schema.proto"
    if not sources or not schema.is_file():
        raise FileNotFoundError("original protocol inputs are incomplete")
    output = output.resolve()
    if output == workspace or workspace in output.parents:
        raise ValueError("generate protocols outside the original source workspace")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([python, "-m", "grpc_tools.protoc", "-I" + str(entropy),
                    "-I" + str(api), "--python_out=" + str(output),
                    "--grpc_python_out=" + str(output),
                    *map(str, sources), str(schema)], check=True)
    # Explicit packages keep these generated modules distinct from unrelated
    # installed packages named proto or galaxy.
    for name in ("proto", "galaxy", "galaxy/kfc_data"):
        (output / name / "__init__.py").touch()
    verification = subprocess.run([
        python, "-c", "\n".join([
            "import sys,json",
            "sys.path.insert(0,sys.argv[1])",
            "from proto import cluster_config_pb2 as cluster",
            "from proto import server_pb2 as server",
            "from proto import learner_pb2,learner_pb2_grpc,controller_pb2,controller_pb2_grpc",
            "from galaxy.kfc_data import embedding_schema_pb2 as schema",
            "value=cluster.ClusterConfig(peer_id=2)",
            "assert cluster.ClusterConfig.FromString(value.SerializeToString()).peer_id==2",
            "assert server.CreateSparseTableRequest.DESCRIPTOR.fields_by_name['sparse_dim']",
            "assert learner_pb2_grpc.LearnerStub and controller_pb2_grpc.EntropyControllerStub",
            "assert schema.BinaryEmbeddingSchema(value_size=64).value_size==64",
            "print(json.dumps(dict(cluster_roundtrip=True,sparse_table_protocol=True,learner_and_controller_stubs=True,embedding_schema=True)))",
        ]), str(output)], check=True, capture_output=True, text=True)
    report = dict(
        source_sha256={p.relative_to(workspace).as_posix(): digest(p) for p in [*sources, schema]},
        generated_sha256={p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*_pb2*.py"))},
        verified=json.loads(verification.stdout),
        original_framework_imported=False,
    )
    (output / "protocol-generation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(prepare(args.workspace, args.output, args.python)))


if __name__ == "__main__":
    main()
