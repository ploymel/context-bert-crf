# Context-BERT-CRF

## To use this code
1. `pip install -r requirements.txt`
2. Add data to `data` for the data format see README.md in `data`
3. `cd src`
4. `chmod +x run.sh` then run `./run.sh`

## The DA in this code consists only 10 tags:
- Feedback
- Statement
- Commissive
- Directive
- PropQ
- SetQ
- ChioceQ
- Salutation
- Apology
- Thanking

To change these DA tags, these 2 snippets of code in both `data_utils.py` and `train.py` are needed to be changed.
```python
# data_utils.py
class DialogueActTaggingDataset(Dataset):
    # other codes before
    
    def _to_label(self, tag):
        DA_TAGS = ['Feedback', 'Commissive', 'Directive', 'Statement', 'PropQ', 'SetQ', 'ChoiceQ', 'Salutation', 'Apology', 'Thanking']
        return DA_TAGS.index(tag)
```

```python
def main():
    def str2int(v):
        global target_names
        if v == 'da':
            target_names = ['Feedback', 'Commissive', 'Directive', 'Statement', 'PropQ', 'SetQ', 'ChoiceQ', 'Salutation', 'Apology', 'Thanking']
            return len(target_names)
        else:
            raise argparse.ArgumentTypeError('Invalid value! Please choose between [semantic, general, som].')
```
