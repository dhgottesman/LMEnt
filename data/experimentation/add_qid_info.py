import json
import click
from json_writer import JsonWriter

def try_get_qid(s: str, qids: dict, lower_case: bool) -> str:
    s = s.lower() if lower_case else s
    return qids.get(s, "NA")

@click.command()
@click.option('--db-file', type=click.Path(exists=True), required=True)
@click.option('--qid-file', type=click.Path(exists=True), required=True)
@click.option('--lower-case/--no-lower-case', default=True)
@click.option('--out-base-path', type=click.Path(exists=False), required=True)
def main(db_file: str, qid_file: str, out_base_path: str, lower_case=True):
    json_writer = JsonWriter(out_base_path, verbose=True)

    with open(qid_file, "r") as f:
        qids = json.load(f)

    f = open(db_file, 'r')
    # while (line := f.readline()) != "":
    for line in f:
        article = json.loads(line)

        article["qid"] = try_get_qid(article['title'], qids, lower_case)

        for i in range(len(article["entities"])):
            article["entities"][i]["qid"] = try_get_qid(article["entities"][i]["entity_name"], qids, lower_case)

        json_writer.write(article)

    json_writer.close()

if __name__ == '__main__':
    main() # type: ignore