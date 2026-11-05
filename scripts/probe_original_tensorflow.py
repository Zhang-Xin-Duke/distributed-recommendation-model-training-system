#!/usr/bin/env python3
"""Probe the original SDK's C API, optional op library and real GPU execution.

This validates native loading and a GPU MatMul with runtime-fed inputs. Full
framework reader, distributed training and control validation run separately.
"""
import argparse
import ctypes as c
import json
from pathlib import Path


class Output(c.Structure):
    _fields_ = [("oper", c.c_void_p), ("index", c.c_int)]


class Buffer(c.Structure):
    _fields_ = [("data", c.c_void_p), ("length", c.c_size_t), ("deallocator", c.c_void_p)]


def registered_operations(data):
    """Read the names from the C API's serialized OpList without a Python SDK."""
    def integer(blob, offset):
        value, shift = 0, 0
        while offset < len(blob) and shift < 64:
            byte = blob[offset]
            value |= (byte & 127) << shift
            offset += 1
            if byte < 128:
                return value, offset
            shift += 7
        raise ValueError("invalid OpList varint")

    offset, names = 0, []
    while offset < len(data):
        tag, offset = integer(data, offset)
        if tag != 10:  # OpList.op: repeated OpDef
            raise ValueError("unexpected OpList field")
        length, offset = integer(data, offset)
        end = offset + length
        if end > len(data):
            raise ValueError("truncated OpDef")
        definition = data[offset:end]
        name_tag, start = integer(definition, 0)
        name_length, start = integer(definition, start)
        if name_tag != 10 or start + name_length > len(definition):
            raise ValueError("invalid OpDef name")
        names.append(definition[start:start + name_length].decode())
        offset = end
    return names


def probe(framework, nccl, tensorflow, ops=None, row_segments=False):
    handles = [c.CDLL(str(path.resolve()), mode=c.RTLD_GLOBAL) for path in (nccl, framework)]
    tf = c.CDLL(str(tensorflow.resolve()), mode=c.RTLD_GLOBAL)
    ptr = c.c_void_p

    def bind(name, result, *arguments):
        function = getattr(tf, name)
        function.restype, function.argtypes = result, list(arguments)
        return function

    version = bind("TF_Version", c.c_char_p)().decode()
    status = bind("TF_NewStatus", ptr)()
    get_code = bind("TF_GetCode", c.c_int, ptr)
    message = bind("TF_Message", c.c_char_p, ptr)

    def check():
        if get_code(status):
            raise RuntimeError(message(status).decode())

    core_buffer = bind("TF_GetAllOpList", c.POINTER(Buffer))()
    core = core_buffer.contents
    if not 0 < core.length <= 64 << 20:
        raise RuntimeError("original SDK returned an invalid core OpList")
    core_operations = registered_operations(c.string_at(core.data, core.length))
    bind("TF_DeleteBuffer", None, c.POINTER(Buffer))(core_buffer)
    if not {"MatMul", "Placeholder"}.issubset(core_operations):
        raise RuntimeError("original SDK is missing required core operations")
    library, operation_names = None, []
    if ops:
        library = bind("TF_LoadLibrary", ptr, c.c_char_p, ptr)(str(ops.resolve()).encode(), status)
        check()
        definitions = bind("TF_GetOpList", Buffer, ptr)(library)
        if not 0 < definitions.length <= 16 << 20:
            raise RuntimeError("custom library did not provide a bounded, nonempty OpList")
        operation_names = registered_operations(c.string_at(definitions.data, definitions.length))
        if not operation_names:
            raise RuntimeError("custom library did not register operations")
    graph = bind("TF_NewGraph", ptr)()
    new_op = bind("TF_NewOperation", ptr, ptr, c.c_char_p, c.c_char_p)
    attr_type = bind("TF_SetAttrType", None, ptr, c.c_char_p, c.c_int)
    attr_shape = bind("TF_SetAttrShape", None, ptr, c.c_char_p, c.POINTER(c.c_int64), c.c_int)
    finish = bind("TF_FinishOperation", ptr, ptr, ptr)
    set_device = bind("TF_SetDevice", None, ptr, c.c_char_p)
    shape = (c.c_int64 * 2)(2, 2)
    placeholders = []
    for name in (b"a", b"b"):
        op = new_op(graph, b"Placeholder", name)
        attr_type(op, b"dtype", 1)  # TF_FLOAT
        attr_shape(op, b"shape", shape, 2)
        placeholders.append(Output(finish(op, status), 0))
        check()
    op = new_op(graph, b"MatMul", b"gpu_product")
    set_device(op, b"/device:GPU:0")
    add_input = bind("TF_AddInput", None, ptr, Output)
    for value in placeholders:
        add_input(op, value)
    attr_type(op, b"T", 1)
    attr_bool = bind("TF_SetAttrBool", None, ptr, c.c_char_p, c.c_ubyte)
    for name in (b"transpose_a", b"transpose_b"):
        attr_bool(op, name, 0)
    product = Output(finish(op, status), 0)
    check()
    row_inputs, row_product = [], None
    if row_segments:
        if "GpuRowLengthSegment" not in operation_names:
            raise RuntimeError("row-segment execution requires the original GPU operation library")
        for name, dimensions in [(b"row_ends", [-1]), (b"row_values_size", [])]:
            op = new_op(graph, b"Placeholder", name)
            attr_type(op, b"dtype", 3)  # TF_INT32
            dimensions_array = (c.c_int64 * len(dimensions))(*dimensions)
            attr_shape(op, b"shape", dimensions_array, len(dimensions))
            row_inputs.append(Output(finish(op, status), 0))
            check()
        op = new_op(graph, b"GpuRowLengthSegment", b"original_gpu_row_segments")
        set_device(op, b"/device:GPU:0")
        for value in row_inputs:
            add_input(op, value)
        attr_type(op, b"T", 3)
        attr_type(op, b"Tindicessize", 3)
        row_product = Output(finish(op, status), 0)
        check()
    options = bind("TF_NewSessionOptions", ptr)()
    # ConfigProto.gpu_options.allow_growth=true. Soft placement remains false;
    # an unavailable GPU must fail rather than execute the requested op on CPU.
    configuration = c.create_string_buffer(b"\x32\x02\x20\x01")
    bind("TF_SetConfig", None, ptr, ptr, c.c_size_t, ptr)(options, configuration, 4, status)
    check()
    session = bind("TF_NewSession", ptr, ptr, ptr, ptr)(graph, options, status)
    check()
    devices = bind("TF_SessionListDevices", ptr, ptr, ptr)(session, status)
    check()
    count = bind("TF_DeviceListCount", c.c_int, ptr)(devices)
    device_type = bind("TF_DeviceListType", c.c_char_p, ptr, c.c_int, ptr)
    types = [device_type(devices, index, status).decode() for index in range(count)]
    check()
    if "GPU" not in types:
        raise RuntimeError("original SDK did not expose a GPU")
    allocate = bind("TF_AllocateTensor", ptr, c.c_int, c.POINTER(c.c_int64), c.c_int, c.c_size_t)
    tensor_data = bind("TF_TensorData", ptr, ptr)
    tensors = []
    for _ in range(2):
        tensor = allocate(1, shape, 2, 16)
        c.memmove(tensor_data(tensor), (c.c_float * 4)(1, 2, 3, 4), 16)
        tensors.append(tensor)
    result = ptr()
    run = bind("TF_SessionRun", None, ptr, ptr, c.POINTER(Output), c.POINTER(ptr), c.c_int,
               c.POINTER(Output), c.POINTER(ptr), c.c_int, c.POINTER(ptr), c.c_int, ptr, ptr)
    run(session, None, (Output * 2)(*placeholders), (ptr * 2)(*tensors), 2,
        c.byref(product), c.byref(result), 1, None, 0, None, status)
    check()
    values = list(c.cast(tensor_data(result), c.POINTER(c.c_float * 4)).contents)
    if values != [7.0, 10.0, 15.0, 22.0]:
        raise RuntimeError("GPU matrix result mismatch: " + repr(values))
    row_cases = []
    if row_segments:
        byte_size = bind("TF_TensorByteSize", c.c_size_t, ptr)
        tensor_type = bind("TF_TensorType", c.c_int, ptr)
        for ends, expected in [([2, 3, 5, 8], [0, 0, 1, 2, 2, 3, 3, 3]),
                               ([0, 0, 2, 2, 5, 5], [2, 2, 4, 4, 4])]:
            vector_shape = (c.c_int64 * 1)(len(ends))
            ends_tensor = allocate(3, vector_shape, 1, len(ends) * 4)
            size_tensor = allocate(3, None, 0, 4)
            c.memmove(tensor_data(ends_tensor), (c.c_int32 * len(ends))(*ends), len(ends) * 4)
            c.memmove(tensor_data(size_tensor), c.byref(c.c_int32(len(expected))), 4)
            row_result = ptr()
            run(session, None, (Output * 2)(*row_inputs), (ptr * 2)(ends_tensor, size_tensor), 2,
                c.byref(row_product), c.byref(row_result), 1, None, 0, None, status)
            check()
            if tensor_type(row_result) != 3 or byte_size(row_result) != len(expected) * 4:
                raise RuntimeError("original GPU row output type or size mismatch")
            actual = list(c.cast(tensor_data(row_result), c.POINTER(c.c_int32 * len(expected))).contents)
            if actual != expected:
                raise RuntimeError("original GPU row-segment result mismatch")
            row_cases.append(dict(row_ends=ends, segments=actual))
            tensors.extend([ends_tensor, size_tensor, row_result])
    bind("TF_CloseSession", None, ptr, ptr)(session, status)
    check()
    bind("TF_DeleteSession", None, ptr, ptr)(session, status)
    check()
    for tensor in [*tensors, result]:
        bind("TF_DeleteTensor", None, ptr)(tensor)
    bind("TF_DeleteDeviceList", None, ptr)(devices)
    bind("TF_DeleteSessionOptions", None, ptr)(options)
    bind("TF_DeleteGraph", None, ptr)(graph)
    bind("TF_DeleteStatus", None, ptr)(status)
    if library:
        bind("TF_DeleteLibraryHandle", None, ptr)(library)
    return dict(tensorflow=version, device_types=types, gpu_matmul=values,
                core_operation_count=len(core_operations),
                op_library_loaded=str(ops) if ops else None,
                registered_operation_count=len(operation_names),
                registered_operations=operation_names,
                original_gpu_row_cases=row_cases,
                full_framework_training_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("framework", "nccl", "tensorflow", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ops", type=Path)
    parser.add_argument("--row-segments", action="store_true")
    args = parser.parse_args()
    result = probe(args.framework, args.nccl, args.tensorflow, args.ops, args.row_segments)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
