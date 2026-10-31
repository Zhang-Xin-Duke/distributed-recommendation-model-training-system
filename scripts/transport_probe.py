"""CPU-only framed transfer control for a prepared loopback/forwarded endpoint."""
import argparse
import json
import pathlib
import socket
import statistics
import struct
import time

LIMIT = 16 << 20


def receive(connection, size):
    output = bytearray()
    while len(output) < size:
        part = connection.recv(size-len(output))
        if not part:
            if output:
                raise EOFError("incomplete transport frame")
            return None
        output.extend(part)
    return output


def server(address):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
        listener.listen(1)
        print("READY", flush=True)
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(30)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while (header := receive(connection, 4)) is not None:
                size = struct.unpack("!I", header)[0]
                if not 1 <= size <= LIMIT:
                    raise ValueError("transport frame exceeds budget")
                if receive(connection, size) is None:
                    raise EOFError("missing transfer body")
                connection.sendall(b"x")


def client(address, sizes, repeats):
    output = []
    with socket.create_connection(address, timeout=10) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for size in sizes:
            if not 1 <= size <= LIMIT:
                raise ValueError("invalid transfer size")
            frame = struct.pack("!I", size) + b"x"*size
            elapsed = []
            for repeat in range(repeats+3):
                begin = time.perf_counter()
                connection.sendall(frame)
                if receive(connection, 1) != b"x":
                    raise RuntimeError("transfer was not acknowledged")
                duration = time.perf_counter()-begin
                if repeat >= 3:
                    elapsed.append(duration)
            median = statistics.median(elapsed)
            output.append(dict(payload_bytes=size, repetitions=repeats,
                median_ms=median*1000, minimum_ms=min(elapsed)*1000,
                maximum_ms=max(elapsed)*1000, upload_mib_per_second=size/median/(1<<20)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["server", "client"])
    parser.add_argument("--address", default="127.0.0.1:19200")
    parser.add_argument("--sizes", default="8192,16384,32768,65536,131072,262144,1048576")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    host, port = args.address.rsplit(":",1)
    if not 1 <= args.repeats <= 1000:
        parser.error("repeats must be 1..1000")
    if args.mode == "server":
        server((host,int(port)))
    else:
        result = client((host,int(port)), [int(s) for s in args.sizes.split(",")], args.repeats)
        text = json.dumps(result, indent=2)+"\n"
        if args.out:
            args.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
