import evaluate

# Measure F1 as in the benchmark repo: https://github.com/orai-nlp/BasqueGLUE/blob/main/eval_basqueglue.py


def weighted_f1_score(items):
    f1_metric = evaluate.load("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="weighted")[
        "f1"
    ]
    return f1_score

