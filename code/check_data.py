"""Check locally supplied application CSVs against the recorded source hashes."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data')
    args=parser.parse_args()
    expected=json.loads((ROOT/'data/checksums.json').read_text())
    errors=[]
    for name,value in expected.items():
        path=args.data_root/name
        status='missing' if not path.exists() else 'ok' if hashlib.sha256(path.read_bytes()).hexdigest()==value else 'checksum differs'
        print(name+': '+status)
        if status!='ok':errors.append(name)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
