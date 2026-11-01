"""Start a native GPU owner before the framework initializes its devices/peers."""
import ctypes as c
import json
import os
import pathlib
import sys
import weakref


class _Status(c.Structure):
    _fields_=[('code',c.c_int32),('message',c.c_char*1024)]


_Create=c.CFUNCTYPE(c.c_int,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(_Status))
_Destroy=c.CFUNCTYPE(None,c.c_void_p)


class _ApiPrefix(c.Structure):
    _fields_=[('abi_version',c.c_uint32),('struct_bytes',c.c_uint32),
              ('create',_Create),('destroy',_Destroy)]


def _destroy(library,api,pointer):
    if pointer.value:api.destroy(pointer)


class NativeWorkerOwner:
    def __init__(self,configuration,*,runtime_library=None):
        if sys.platform!='linux':raise RuntimeError('native GPU owner installation requires the Linux static runtime')
        path=pathlib.Path(runtime_library or os.environ['EPS_NATIVE_RUNTIME']).resolve()
        library=c.CDLL(str(path));library.eps_runtime_api_v1.restype=c.POINTER(_ApiPrefix)
        api=library.eps_runtime_api_v1().contents
        if api.abi_version!=1 or api.struct_bytes<c.sizeof(_ApiPrefix):raise RuntimeError('native owner ABI mismatch')
        pointer=c.c_void_p();status=_Status()
        code=api.create(json.dumps(configuration).encode(),c.byref(pointer),c.byref(status))
        if code:raise RuntimeError(status.message.decode())
        try:self._cleanup=weakref.finalize(self,_destroy,library,api,pointer)
        except BaseException:
            api.destroy(pointer);raise

    def close(self):self._cleanup()
