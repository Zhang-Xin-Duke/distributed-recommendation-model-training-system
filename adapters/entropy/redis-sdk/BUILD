cc_library(
    name = "bilibili-redis",
    srcs = ["src/client.cpp"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    copts = ["-std=c++17"],
    deps = ["@brpc//:brpc", "@json//:json"],
    visibility = ["//visibility:public"],
)
