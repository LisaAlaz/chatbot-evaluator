## UniEval-Dialog Evaluation Metric

---

Implementation of an evaluation metric using [UniEval](https://github.com/maszhongming/UniEval)

---

### Quick start

1. Drag and drop this directory into your working `metrics` directory.

2. Download/Clone a UniEval-Dialog model into this directory from https://huggingface.co/MingZhong/unieval-dialog

```
git clone https://huggingface.co/MingZhong/unieval-dialog
```

3. Install requirements

```
pip install -r requirements.txt
```

4. Add the metric into your evaluator. Example in `evaluator.py`:

```
from [Your Directory].metrics.unieval import UniEvalDialogMetric

...

self.metrics = [
    ...
    UniEvalDialogMetric(os.path.join(config['local_dir'], 'metrics/unieval/unieval-dialog'))
]

```