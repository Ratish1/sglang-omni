"""Print a server port P below the ephemeral range whose P and P + 100 bind now and that this queue has not used."""

import socket
import sys

used_path = sys.argv[1]
with open(used_path) as used_file:
    used = {int(word) for word in used_file.read().split()}
for start, end in (
    (20000, 27800),
    (10000, 15800),
    (16200, 19800),
    (28000, 29400),
    (30000, 32600),
):
    for port in range(start, end, 200):
        if port in used:
            continue
        free = True
        for candidate in (port, port + 100):
            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                free = False
            finally:
                probe.close()
        if free:
            with open(used_path, "a") as used_file:
                used_file.write(f"{port}\n")
            print(port)
            sys.exit(0)
sys.exit(1)
