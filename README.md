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
1.  Create the directory `OLMo-core/hp_final`.
2.  Copy the `dataset-cache` directory from [this link](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/tree/main/dataset-cache) into `OLMo-core/hp_final`.
3.  Copy the `dataset-tokenized` directory from [this link](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/tree/main/dataset-tokenized) into the root `LMEnt` directory.
4.  Copy the `dataset-metadata` directory from [this link](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset/tree/main/dataset-metadata) into the root `LMEnt` directory.
5.  Unzip the `dataset-metadata/part-[0-7]-00000.csv.gz` files.

The final directory structure should look like this: 
```
dolma
environment.yml
maverick-coref
olmes
ReFiNED
OLMo-core (submodule)
    > hp_final
        > dataset-cache
README.md
ReFinED
retrieval-index
dataset-metadata
dataset-tokenized
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

1. **Download and Unpack**: Unpack the elasticsearch_enwiki_dump.tar.gz file into a directory (e.g., `/mnt/elasticsearch_lment/`). The unpacked contents will contain the necessary subdirectories.

2. **Configure path.repo**: In `elasticsearch.yml` (sometimes under `elasticsearch-8.13.4/config/elasticsearch.yml`, set the `path.repo` to point to the parent directory of the unpacked contents (e.g., `path.repo: /mnt/elasticsearch_lment/`) and restart Elasticsearch (cd elasticsearch-8.13.4; /bin/elasticsearch).

3. **Register the Repository:**

```
PUT _snapshot/lment
{
  "type": "fs",
  "settings": {
    "location": "elasticsearch_lment" 
  }
}
```

4. **Restoring the Indices**:

```
POST _snapshot/lment/lment/_restore?wait_for_completion=true
{
  "indices": "enwiki", 
  "rename_pattern": "enwiki", 
  "rename_replacement": "lment_enwiki_ci" 
}

POST _snapshot/lment/lment/_restore?wait_for_completion=true
{
  "indices": "enwiki_case_sensitive", 
  "rename_pattern": "enwiki_case_sensitive", 
  "rename_replacement": "lment_enwiki_cs" 
}
```

- `lment_enwiki_cs`: The case sensitive index.
- `lment_enwiki_ci`: The case insensitive index.

Please refer to the README in `LMEnt/retrieval-index` for examples of entity-based retrieval queries.





