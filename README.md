# LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations
This repository contains the official code for the paper: "LMEnt: A Suite for Analyzing Knowledge in Language Models from Pretraining Data to Representations" (2025).

---

## Setup

```
git clone --recurse-submodules git@github.com:dhgottesman/LMEnt.git
cd LMEnt
conda create env -f environment.yml
```

## Pretraining Dataset
The dataset is available on [Hugging Face](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset).

To set up the dataset and dataloader, please follow these steps:
1.  Download the [LMEnt-Dataset](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset) dataset.
2.  Unzip the `LMEnt-Dataset/dataset-metadata/part-[0-7]-00000.csv.gz` files.
3.  Run `setup.py <absolute path to LMEnt-Dataset directory>`

The final directory structure should look like this: 
```
LMEnt
    > dolma
    > environment.yml
    > maverick-coref
    > olmes
    > ReFiNED
    > OLMo-core (submodule)
    > README.md
    > ReFinED
    > retrieval-index
LMEnt-Dataset
    > dataset-cache
    > dataset-metadata
    > dataset-tokenized
```

This table summarizes the disk space requirements:

| Step | Directory/File Set | Context | Compressed Size | Decompressed Size |
| :--- | :--- | :--- | :--- | :--- |
| **2** | `dataset-cache` | Used to build the dataset and dataloader. | **727 MB** | N/A |
| **3** | `dataset-tokenized` | Tokenized and concatendated data chunks. | **15 GB** | N/A |
| **5** | `dataset-metadata` | CSV files entity mention annotations for each data chunk. | **30 GB** | **212 GB** |

## LMEnt Models
The models with checkpoints taken every 10K steps are available in the [Hugging Face Collection](https://huggingface.co/collections/dhgottesman/lment).

The following steps demonstrate how to load a specific model checkpoint, e.g., LMEnt-1B-1E checkpointed at training step 10,000:
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "dhgottesman/LMEnt-1B-1E"       # model root
sub = "step10000"                          # the checkpoint folder

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=sub, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    subfolder=sub
)

inputs = tokenizer("Hello from step10000!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Entity-Based Retrieval Index
LMEnt's entity-based retrieval index is built on top of [Elasticsearch](https://www.elastic.co/elasticsearch).

To build the index from scratch, run the script `retrieval-index/create_es_index.py`. This can take up to a day to build.

You can alternatively download both case-sensitive and case-insensitive pre-built indexes at [LINK].
Please note that this tar file is **127GB**. To register the pre-built files, please follow the following steps:

1. **Download and Unpack**: Unpack the [elasticsearch_lment.tar.gz](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/blob/main/elasticsearch_lment.tar.gz) file into a directory (e.g., `/path/to/elasticsearch_lment`). The unpacked contents contain the `lment` directory with the necessary files to restore the index.

2. **Configure path.repo**: In `elasticsearch.yml` (sometimes under `elasticsearch-8.13.4/config/elasticsearch.yml`, set the `path.repo` to point to the parent directory of the unpacked contents (e.g., `path.repo: /path/to/elasticsearch_lment`) and restart Elasticsearch (cd elasticsearch-8.13.4; /bin/elasticsearch).

3. **Register the Repository:**

```
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X PUT 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment' \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/path/to/elasticsearch_lment/lment"
    }
  }'
```

4. **Restoring the Indices**:

Restore the case-sensitive index.
```
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X POST 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment/lment/_restore?wait_for_completion=true' \
  -d '{
    "indices": "enwiki_case_sensitive",
    "rename_pattern": "enwiki_case_sensitive",
    "rename_replacement": "lment_cs",
    "include_global_state": false
  }'
```

Restore the case-insensitive index.
```
curl -k -u 'elastic:<PASSWORD>' \
  -H 'Content-Type: application/json' \
  -X POST 'https://<ELASTIC_SERVER_ADDR>:<PORT>/_snapshot/lment/lment/_restore?wait_for_completion=true' \
  -d '{
    "indices": "enwiki",
    "rename_pattern": "enwiki",
    "rename_replacement": "lment_ci",
    "include_global_state": false
  }'
```

> *Note that case in/sensitivity is relevant only if you want to perform string-based retrieval.
We include examples of entity-based retrieval and string-based retrieval queries in the README under `retrieval-index`.*

## Analyzing Learning Dynamics
If you want to analyze learning dynamics on PopQA, you can use our [annotated dataset](https://huggingface.co/datasets/dhgottesman/popqa-kas). It includes precomputed chunk identifiers `chunk_id` from the pretraining corpus that mention the subject entity `subject_chunks`, the answer entity `answer_chunks`, and co-occuring `shared_chunks`.

**For this analysis, you don't need to download the entire pretraining dataset, you only need** `LMEnt-Dataset/dataset-cache/batch_indices.npy`.

To map a given chunk ID to the training step at which a model **trained for one epoch** saw that chunk, you can use the following code:
```
import numpy as np

batch_indices = np.load(
    "LMEnt-Dataset/dataset-cache/batch_indices.npy",
    allow_pickle=True
)

# Example chunk identifier
chunk_id = 2955255

for step, chunk_ids_in_batch in enumerate(batch_indices):
    if chunk_id in chunk_ids_in_batch:
        print(f"Chunk with chunk_id {chunk_id} was seen in training step {step}")

> Chunk with chunk_id 2955255 was seen in training step 16618
```

## Training LMEnt Models
To kick off training LMEnt models, run `src/examples/kas/train.py <path/to/config.json>`. You need to define a `config.json` like `src/examples/kas/kas_config.json`.

## Annotating Pretraining Data
### ReFinED
Follow the instructions in this [README](https://github.com/dhgottesman/ReFinED/blob/main/README.md) to process a raw Wikipedia dump, extract hyperlinks, and generate all required files for entity linking.

Run `ReFinED/run_refined.py` with slurm using `ReFinED/run.slurm`.

### Maverick
Run `maverick-coref/run_maverick.py`with slurm using `maverick-coref/run.slurm`.

## Tokenizing Pretraining Data
Run `dolma/python/run.slurm`. TBD: Changes to merge the ReFinED and Maverick annotations need to be added.
