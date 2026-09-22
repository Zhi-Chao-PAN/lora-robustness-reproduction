"""Download official PAWS-Wiki archive; retain only its public test for stress evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

URL = 'https://storage.googleapis.com/paws/english/paws_wiki/labeled_final.tar.gz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    archive = args.out / 'labeled_final.tar.gz'
    with urllib.request.urlopen(URL, timeout=90) as response, archive.open('wb') as target:
        while block := response.read(1024 * 1024):
            target.write(block)
    with tarfile.open(archive) as source:
        matches = [entry for entry in source.getmembers() if entry.name.endswith('/test.tsv') and entry.isfile()]
        if len(matches) != 1:
            raise ValueError('Unexpected official archive structure')
        member = matches[0]
        data = source.extractfile(member).read()
    (args.out / 'test.tsv').write_bytes(data)
    records = {'source_url': URL, 'source_repository': 'https://github.com/google-research-datasets/paws', 'paper': 'https://arxiv.org/abs/1904.01130', 'archive_member': member.name, 'only_split_extracted': 'test', 'training_or_development_used': False, 'sha256': {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest() for name in ['labeled_final.tar.gz', 'test.tsv']}}
    (args.out / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
