from datasets import load_dataset
import sys
import time
import os
from pathlib import Path



# Task 3
dataset3 = load_dataset("stanfordnlp/imdb", split="train")

dataset3.to_csv("imdb_train.csv")
dataset3.to_parquet("imdb_train.parquet")
json_size = os.path.getsize("imdb_train.csv")
parquet_size = os.path.getsize("imdb_train.parquet")

print(f"json file is {json_size} bytes")

print(f"parquet file is {parquet_size} bytes")

# Task 4
split = dataset3.train_test_split(test_size=0.15, seed=42)
train_val = split["train"].train_test_split(test_size=0.1765, seed=42)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
# Task 1

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train",streaming=True)

print("Examples of the dataset:")

for i, sentence in enumerate(dataset.take(5)): #for i in range(5): Wont work well because we do streaming

    print(f"premise: {sentence['sentence1']}")
    print(f"hypothesis: {sentence['sentence2']}")
    print(f"Label: {sentence['label']}")
    print("-" * 50)

# Task 2
def c4_task(duration=10):
    en = load_dataset("allenai/c4", "en",streaming=True,split="train")
    data_itr = iter(en)
    count = 0
    start_time = time.time()
    end_time = start_time + duration
    while True:
        current_time = time.time()
        if current_time >= end_time:
            break
        try:
            item  = next (data_itr)
            count += 1
        except StopIteration:
            print("\nreached the end!")
            break
    print("\n--- Benchmark Complete ---")

    print(f"Total processed examples: {count:,}")
    print(f"Average output: {round(count / duration, 2)} examples/sec")

c4_task(duration=10)
