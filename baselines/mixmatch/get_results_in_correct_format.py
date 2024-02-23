import argparse
import json
import os
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--samples_file', type=str, required=True)
    parser.add_argument('--target_label', type=str, required=True)

    args = parser.parse_args()

    out_dirname = args.target_label + '_results'
    out_dirname = os.path.join(
        out_dirname, datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    os.makedirs(out_dirname, exist_ok=True)

    outfile_name = os.path.join(out_dirname, args.target_label + '.jsonl')

    with open(os.path.join(out_dirname, 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    with open(args.input_file, 'r') as f:
        input_lines = [x.strip().split('\t') for x in f.readlines()]

    with open(args.samples_file, 'r') as f:
        samples = [x.strip() for x in f.readlines()]

    with open(outfile_name, 'w') as f:
        for (_, _, original), sample in zip(input_lines, samples):
            print(f'{original}\t{sample}')

            retval = {}
            retval['target_label'] = args.target_label
            retval['input_label'] = args.input_file
            retval['original_text'] = original
            retval['paraphrase'] = None
            retval['decoded'] = [sample]

            f.write(json.dumps(retval) + '\n')
