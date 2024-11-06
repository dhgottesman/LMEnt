import os
import json
import click
import ijson
from numpy import require
from tqdm import tqdm
from qwikidata.entity import WikidataItem
from qwikidata.json_dump import WikidataJsonDump

def get_enwiki_title(ed) -> str:
    """Get english language wikipedia page title."""
    if (
        isinstance(ed["sitelinks"], dict)
        and "enwiki" in ed["sitelinks"]
    ):
        return ed["sitelinks"].get("enwiki", {}).get("title", "")

    return ""

def iterate(path, start=None, end=None):
    prev_tell = start or 0
    with open(path, mode='rb') as f:
        if start:
            f.seek(start)
            # Skip to the next line if we're not at the start of a line
            first_line = f.readline().decode('utf-8').rstrip(",\n")
            try:
                yield json.loads(first_line), f.tell() - prev_tell
                prev_tell = f.tell()
            except json.JSONDecodeError:
                pass

        for linebytes in f:
            line = linebytes.decode('utf-8').rstrip(",\n")
            if line in ["[", "]"]:
                continue

            tell = f.tell()
            yield json.loads(line), tell - prev_tell

            prev_tell = tell
            if end and tell >= end:
                break

def get_title_qid_mapping(file_path, start=None, end=None):
    size = (end - start if end else os.path.getsize(file_path)) / 1024**2
    pbar = tqdm(total=size, unit='MB')

    for ed, read in iterate(file_path, start, end):
        if ed["type"] == "item":
            entity_id = ed["id"]
            title = get_enwiki_title(ed)
            if title:
                yield title, entity_id
        
        pbar.update(read / 1024**2)

@click.command()
@click.option('--file-path', type=click.Path(exists=True), required=True)
@click.option('--out-path', type=click.Path(exists=False), required=True)
@click.option('--start', type=int, default=None)
@click.option('--end', type=int, default=None)
def main(file_path, out_path, start=None, end=None):
    assert os.path.exists(file_path), f"File path does not exist: {file_path}"
    assert not os.path.exists(out_path), f"Output path already exists: {out_path}"
    assert (start is None and end is None) or (start is not None and end is not None), "Both start and end must be provided"

    with open(out_path, 'w') as writer:
        try:
            writer.write("{\n")
            for title, qid in get_title_qid_mapping(file_path, start, end):
                writer.write(f'\t"{title}": "{qid}",\n')
        except (Exception, KeyboardInterrupt) as e:
            print(e)
        finally:
            writer.write("}\n")

if __name__ == '__main__':
    main()
