#!/usr/bin/env python3
"""Bind a private original-framework workspace to installed Linux CUDA tools."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def command(*args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, check=True, **kwargs)


def compiler_includes(compiler):
    text = command(compiler, "-E", "-x", "c++", "-", "-v", input="").stderr
    match = re.search(r"#include <\.\.\.> search starts here:\n(.*?)End of search list\.", text, re.S)
    if not match:
        raise RuntimeError("compiler did not report its actual system include paths")
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def configure(workspace, cuda, gcc, clang=None):
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("original native build configuration requires Linux x86_64")
    workspace, cuda = workspace.resolve(strict=True), cuda.resolve(strict=True)
    if not (workspace / "bundle-manifest.json").is_file():
        raise ValueError("configure an independent materialized bundle")
    for name in ("bin/nvcc", "bin/bin2c", "bin/fatbinary", "bin/nvlink", "bin/crt/link.stub", "include/cuda.h"):
        if not (cuda / name).is_file():
            raise FileNotFoundError("missing actual CUDA input: " + name)
    header = (cuda / "include/cuda.h").read_text()
    value = int(re.search(r"^#define\s+CUDA_VERSION\s+(\d+)", header, re.M).group(1))
    cuda_version = f"{value // 1000}.{(value % 1000) // 10}"
    expected_header = workspace / "repositories/tensorflow__2.12.1-nvidia/include/third_party/gpus/cuda/cuda_config.h"
    expected = re.search(r'#define TF_CUDA_VERSION "([^"]+)"', expected_header.read_text()).group(1)
    if cuda_version != expected:
        raise ValueError(f"installed CUDA {cuda_version} differs from SDK CUDA {expected}")
    gcc = str(Path(shutil.which(gcc) or gcc).resolve(strict=True))
    includes = {"gcc": compiler_includes(gcc)}
    clang_bases, clang_resources = {}, {}
    if clang:
        clang = str(Path(shutil.which(clang) or clang).resolve(strict=True))
        includes["clang"] = compiler_includes(clang)
        clang_resources["clang"] = command(clang, "-print-resource-dir").stdout.strip()
        clang_bases["clang"] = str(Path(clang).parent.parent)
    system = workspace / ".resources/toolchains/.system/cuda"
    system.mkdir(parents=True, exist_ok=True)
    link = system / "cuda"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError("CUDA fixture already contains a real source directory")
    link.symlink_to(cuda, target_is_directory=True)
    # This reviewed helper only emits BUILD rules from actual toolkit files and
    # reads ELF SONAMEs with objdump. The original bootstrap is not invoked.
    generator = (workspace / ".resources/scripts/__generate_cuda_build.sh").read_text()
    # Formatting is optional; rule generation and ELF inspection still run with
    # their original error handling when buildifier is not installed.
    generator = generator.replace('command "buildifier" "${CUDA_BUILD}"',
                                  'if command -v buildifier >/dev/null; then buildifier "${CUDA_BUILD}"; fi')
    command("bash", "-s", "--", str(system), input=generator)
    facts = dict(WORKSPACE_PATH=str(workspace), ACTION_PATH=os.environ["PATH"],
                 NPROC=len(os.sched_getaffinity(0)), OS=platform.freedesktop_os_release(),
                 SYSTEM_CXX_INCLUDES=includes, ENABLE_LIBCPP=0,
                 CLANG_BASES=clang_bases, CLANG_RESOURCES=clang_resources,
                 GCC_EXECUTABLES={"gcc": gcc},
                 CUDA_HOMES={".": dict(INSTALL=str(cuda),
                                      PACKAGE=system.relative_to(workspace).as_posix(),
                                      VERSION=cuda_version)})
    (workspace / ".resources/rules/_environment.bzl").write_text(
        "# Actual Linux compiler and CUDA installation facts.\n" +
        "\n".join(key + " = " + repr(value) for key, value in facts.items()) + "\n")

    # Keep the original public target names while enumerating headers from the
    # installed RDMA development package, rather than an old machine's file list.
    rdma = workspace / "repositories/infiniband"
    include = Path("/usr/include/infiniband")
    if not (include / "verbs.h").is_file():
        raise FileNotFoundError("install libibverbs development headers")
    (rdma / "include").mkdir(exist_ok=True)
    target = rdma / "include/infiniband"
    if target.is_symlink():
        target.unlink()
    target.symlink_to(include, target_is_directory=True)
    (rdma / "lib").mkdir(exist_ok=True)
    cache = command("ldconfig", "-p").stdout
    imports = []
    for name in ("ibverbs", "mlx5", "mlx4", "efa"):
        soname = "lib" + name + ".so.1"
        match = re.search(r"\s" + re.escape(soname) + r"\s.*=>\s+(\S+)", cache)
        if not match:
            if name in {"ibverbs", "mlx5"}:
                raise FileNotFoundError("install actual RDMA library " + soname)
            continue
        actual = Path(match.group(1)).resolve(strict=True)
        target = rdma / "lib" / soname
        if target.is_symlink():
            target.unlink()
        target.symlink_to(actual)
        deps = '[":api"]' if name == "ibverbs" else '[":api", ":ibverbs"]'
        imports.append(f'cc_import(name="{name}", shared_library="lib/{soname}", deps={deps})')
    (rdma / "BUILD").write_text(
        'package(default_visibility=["//visibility:public"])\n'
        'cc_library(name="api", hdrs=glob(["include/infiniband/**/*.h"]), includes=["include"])\n' +
        "\n".join(imports) + '\nalias(name="infiniband",actual=":ibverbs")\n')
    workspace_file = workspace / "WORKSPACE"
    workspace_text = workspace_file.read_text()
    if "bazel_features_deps()" not in workspace_text:
        workspace_file.write_text(workspace_text +
                                  '\nload("@bazel_features//:deps.bzl", "bazel_features_deps")\n'
                                  'bazel_features_deps()\n')
    macro = workspace / "entropy/contrib/tensorswitch.bzl"
    macro_text = macro.read_text()
    # These options were encoded for a TensorFlow compiler wrapper. The recovered
    # Bazel CUDA toolchain invokes nvcc directly and needs its native spellings.
    cuda_options = {"-nvcc_options=relaxed-constexpr": "--expt-relaxed-constexpr",
                    "-nvcc_options=ftz=true": "--ftz=true"}
    for legacy, native in cuda_options.items():
        macro_text = macro_text.replace('"' + legacy + '"', '"' + native + '"')
    macro.write_text(macro_text)
    optimizer = workspace / "entropy/tensorswitch/ops/mio_adam_op.cc"
    optimizer_text = optimizer.read_text()
    helpers = ["MaybeLockVariableInputMutexesInOrder", "GetInputTensorFromVariable"]
    # C++17 requires these explicit template names to be declared before ADL.
    # Qualify the existing calls to their SDK namespace, preserving the optimizer.
    optimizer_text = re.sub(r"(?<![A-Za-z0-9_:])(" + "|".join(helpers) + r")(?=<)",
                            r"tensorflow::\1", optimizer_text)
    optimizer.write_text(optimizer_text)
    redis_build = workspace / "repositories/common/base/BUILD"
    redis_text = redis_build.read_text()
    marker = '    name = "redis",\n'
    definition = '    defines = ["AI_BUILD_WITH_BAZEL=1"],\n'
    # The original __identity_ai configuration sets this define for all target
    # and host compilations. It selects both Redis includes and Peta's glog
    # implementation. A Redis-only define misses independent Peta consumers.
    if marker + definition in redis_text:
        redis_build.write_text(redis_text.replace(marker + definition, marker, 1))
    result = dict(cuda_version=cuda_version, toolkit_nvcc_version=command(str(cuda / "bin/nvcc"), "--version").stdout,
                  gcc=command(gcc, "--version").stdout.splitlines()[0],
                  clang=command(clang, "--version").stdout.splitlines()[0] if clang else None,
                  configured_toolkit=str(cuda), rdma_imports=len(imports),
                  cuda_option_translation=cuda_options,
                  qualified_tensorflow_helpers=helpers,
                  required_bazel_copts=["-DAI_BUILD_WITH_BAZEL=1"],
                  required_bazel_host_copts=["-DAI_BUILD_WITH_BAZEL=1"],
                  required_bazel_cxxopts=["-fsized-deallocation"],
                  required_bazel_host_cxxopts=["-fsized-deallocation"],
                  required_bazel_defines=["enable_gpu=1"], native_compiled=False)
    (workspace / "linux-configuration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--clang")
    args = parser.parse_args()
    print(json.dumps(configure(args.workspace, args.cuda, args.gcc, args.clang)))


if __name__ == "__main__":
    main()
