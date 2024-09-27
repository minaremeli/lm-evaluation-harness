import random

import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [doc["Accepted Answer"], doc["Worst Answer"]]

        random.shuffle(choices)
        correct_answer_index = choices.index(doc["Accepted Answer"])

        out_doc = {
            "Title": doc["Question Title"],
            "Body": doc["Question Body"],
            "A": choices[0],
            "B": choices[1],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
