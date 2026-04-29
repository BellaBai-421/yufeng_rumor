import json

path = "output/punishment/train.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data):
    text = item.get("rumorText", "")
    if not isinstance(text, str) or not text.strip():
        print("空文本:", i, item)
    if len(text.strip()) <= 3:
        print("过短文本:", i, repr(text), item.get("result"))