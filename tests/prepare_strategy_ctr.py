"""Convert a chronological Criteo TSV prefix to paired training and validation input."""
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, choices=(2, 4, 6), required=True)
    parser.add_argument('--global-batch', type=int, default=192)
    parser.add_argument('--train-batches', type=int, default=128)
    parser.add_argument('--validation-rows', type=int, default=4096)
    args = parser.parse_args()
    if (args.global_batch < args.workers or args.global_batch % args.workers
            or args.train_batches < 1 or args.validation_rows < 1):
        parser.error('use positive sizes and a global batch divisible by the Worker count')
    if args.input.resolve() == args.output.resolve():
        parser.error('output must differ from input')
    train_rows = args.global_batch * args.train_batches
    count = train_rows + args.validation_rows
    digest = hashlib.sha256()
    examples = []
    with args.input.open('rb') as source:
        for index, line in enumerate(islice(source, count)):
            digest.update(line)
            columns = line.rstrip(b'\r\n').split(b'\t')
            if len(columns) != 40 or columns[0] not in (b'0', b'1'):
                parser.error(f'invalid Criteo row {index + 1}')
            keys = []
            for field, value in enumerate(columns[14:]):
                token = bytes([field]) + value
                hashed = hashlib.blake2b(token, digest_size=8).digest()
                keys.append(11 + int.from_bytes(hashed, 'big') % (32768 - 11))
            examples.append({'keys': keys, 'label': int(columns[0])})
    if len(examples) != count:
        parser.error(f'expected {count} rows, got {len(examples)}')
    local = args.global_batch // args.workers
    batches = [[] for _ in range(args.workers)]
    for start in range(0, train_rows, args.global_batch):
        for worker, stream in enumerate(batches):
            begin = start + worker * local
            stream.append(examples[begin:begin + local])
    data = {'batches': batches, 'validation': examples[train_rows:],
            'provenance': {'format': 'criteo-40-column-tsv', 'prefix_sha256': digest.hexdigest(),
                           'train_rows': train_rows, 'validation_rows': args.validation_rows,
                           'global_batch': args.global_batch,
                           'encoding': 'field-index + raw category, BLAKE2b-64, 32757 buckets'}}
    args.output.write_text(json.dumps(data, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
