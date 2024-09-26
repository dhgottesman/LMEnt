import ijson

def get_title_qid_mapping(file_path):
    title_qid_mapping = {}

    with open(file_path, 'r') as reader:
        for item in ijson.items(reader, 'item'):
            qid = item['id']

            if qid == "Q640":
                continue

            # title = item.get('sitelinks', {}).get('enwiki', {}).get('title')
            labels = item['labels']
            if "en" in labels:
                lang = "en"
            elif "en-us" in labels:
                lang = "en-us"
            else:
                continue

            title = labels[lang]["value"]
            title_qid_mapping[title] = qid

    return title_qid_mapping