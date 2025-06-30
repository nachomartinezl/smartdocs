import json, random
from collections import defaultdict
from pathlib import Path

EXAMPLES_PATH = Path("dataset/examples.jsonl")
TRAIN_OUT     = Path("dataset/train_examples.jsonl")
TEST_OUT      = Path("dataset/test_examples.jsonl")
TRAIN_SPLIT   = 0.8
SEED          = 42

random.seed(SEED)

# Read all examples
lines = EXAMPLES_PATH.read_text(encoding="utf-8").splitlines()
examples = [json.loads(line) for line in lines]

# Group by label
by_label = defaultdict(list)
for ex in examples:
    by_label[ex["label"]].append(ex)

train_set, test_set = [], []

for label, items in by_label.items():
    random.shuffle(items)
    split_idx = int(len(items) * TRAIN_SPLIT)
    train_set.extend(items[:split_idx])
    test_set.extend(items[split_idx:])

print(f"✅ Train: {len(train_set)} | Test: {len(test_set)}")

# Write outputs
with TRAIN_OUT.open("w", encoding="utf-8") as f:
    for ex in train_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with TEST_OUT.open("w", encoding="utf-8") as f:
    for ex in test_set:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("🚀 Split done. Files written:")
print(f"- {TRAIN_OUT}")
print(f"- {TEST_OUT}")
