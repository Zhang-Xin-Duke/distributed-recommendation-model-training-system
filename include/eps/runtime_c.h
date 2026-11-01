#ifndef EPS_RUNTIME_C_H
#define EPS_RUNTIME_C_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define EPS_RUNTIME_ABI 1
typedef struct {int32_t code;char message[1024];} EpsStatus;
typedef struct {
  uint32_t table,feature,pooling;
  const uint64_t* keys;uint64_t key_count;
  const uint64_t* offsets;uint64_t offset_count;
  int32_t has_padding;uint64_t padding_key;
} EpsFeature;
typedef struct {
  uint32_t feature;uint64_t rows,columns;
  void* data;int32_t device;void* owner;
  void (*release)(void*);
  int (*wait)(void*,void*,EpsStatus*);
} EpsBuffer;
typedef struct {uint64_t handle,count;EpsBuffer* buffers;void* owner;void (*release)(void*);} EpsBatch;
typedef struct {const char* data;uint64_t size;void* owner;void (*release)(void*);} EpsText;
typedef struct {
  uint32_t abi_version,struct_bytes;
  int (*create)(const char*,void**,EpsStatus*);
  void (*destroy)(void*);
  int (*pull)(void*,const EpsFeature*,uint64_t,int32_t,int32_t,EpsBatch*,EpsStatus*);
  int (*prepare)(void*,const EpsFeature*,uint64_t,uint64_t*,EpsStatus*);
  int (*pull_prepared)(void*,uint64_t,int32_t,int32_t,EpsBatch*,EpsStatus*);
  int (*push)(void*,uint64_t,const EpsBuffer*,uint64_t,void*,EpsStatus*);
  int (*command)(void*,const char*,const char*,EpsText*,EpsStatus*);
} EpsRuntimeApi;
typedef const EpsRuntimeApi* (*EpsRuntimeEntry)(void);
#if defined(_WIN32)
#define EPS_RUNTIME_EXPORT __declspec(dllexport)
#else
#define EPS_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif
EPS_RUNTIME_EXPORT const EpsRuntimeApi* eps_runtime_api_v1(void);
#ifdef __cplusplus
}
#endif
#endif
