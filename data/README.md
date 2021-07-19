# Data
The whole conversation with .csv format is needed.

The data should be added in this format:
`data/da/{dataset_name}/{conversation_file}.csv`

# Conversation file format
Two columns are required (Message, Tag) no need to add header. Is there any header exist, the following line of code need to be changed.

```python
# line 180 in data_utils.py
df = pd.read_csv(dial_file, header=None) # will need to be changed to 
df = pd.read_csv(dial_file)
```
