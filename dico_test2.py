import json

with open("./test2.json", "r", encoding="utf-8") as f:
    dico_str = f.read()
    if not dico_str:
        print("No string!")
    else:
        print("There's a string!")
        dico = json.loads(dico_str)
        print(dico)