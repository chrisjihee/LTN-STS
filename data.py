import json
from pathlib import Path

from datasets import load_dataset


def data_to_classification(data, outdir):
    outdir = Path(outdir)
    print()
    print('=' * 120)
    print(f"* Data: {data}")
    print(f"* Outdir: {outdir}")
    for split in data:
        converted = [
            {
                'guid': example['guid'],
                'sentence1': example['sentence1'],
                'sentence2': example['sentence2'],
                'label': example['labels']['binary-label']
            } for example in data[split]
        ]
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{split}.json"
        print(f"  - {outfile}: {len(converted)}")
        with outfile.open('w') as out:
            json.dump({"version": f"datasets_1.0", "data": converted}, out, ensure_ascii=False, indent=4)
    print('=' * 120)


# load raw datasets
raw_datasets = load_dataset("json", data_files={
    "train": "data/klue-sts/train.json",
    "valid": "data/klue-sts/valid.json"}, field="data")

# save converted datasets
data_to_classification(raw_datasets, "data/klue-sts-cls")
