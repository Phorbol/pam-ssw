"""LASP-side private client; writes the frozen E + 3N force + 9 tensor layout."""
import json
import os
import socket
from pathlib import Path


def main():
    target = Path("external.ene")
    target.unlink(missing_ok=True)
    request = {"coord": Path("external.coord").read_text()}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(30)
        sock.connect(os.environ["LASP_MACE_SOCKET"])
        stream = sock.makefile("rwb")
        stream.write((json.dumps(request) + "\n").encode())
        stream.flush()
        response = json.loads(stream.readline())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "missing external E/F/stress"))
    forces = response["forces"]
    tensor = response["stress_wire"]
    if len(forces) != 4 or any(len(row) != 3 for row in forces) or len(tensor) != 9:
        raise ValueError("response violates frozen Cu4 E/F/full-tensor layout")
    content = [f"{float(response['energy']):.17g}"]
    content.extend(" ".join(f"{float(value):.17g}" for value in row) for row in forces)
    content.append(" ".join(f"{float(value):.17g}" for value in tensor))
    tmp = Path("external.ene.tmp")
    tmp.write_text("\n".join(content) + "\n")
    tmp.replace(target)


if __name__ == "__main__":
    main()
