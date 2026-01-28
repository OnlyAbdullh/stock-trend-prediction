import json

with open("temp/nt.json", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("restored_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
