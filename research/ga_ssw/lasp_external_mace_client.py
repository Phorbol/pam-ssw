"""LASP external callback; stdlib only, calculator stays in the host process."""
import json,os,socket
from pathlib import Path


def main():
    target=Path('external.ene')
    target.unlink(missing_ok=True)  # A failed call must not leave a stale response.
    request={'coord':Path('external.coord').read_text()}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
        sock.settimeout(120);sock.connect(os.environ['LASP_MACE_SOCKET'])
        stream=sock.makefile('rwb');stream.write((json.dumps(request)+'\n').encode());stream.flush()
        response=json.loads(stream.readline())
    if not response.get('ok'):raise RuntimeError(response.get('error','missing external result'))
    content=f"{response['energy']:.17g}\n"+'\n'.join(' '.join(f'{v:.17g}' for v in row) for row in response['forces'])+'\n'
    temp=Path('external.ene.tmp');temp.write_text(content);temp.replace(target)


if __name__=='__main__':main()
