import json

#dico_str = '{"stuff": ["more stuff", "even more stuff"], "bruh": 1.5}'
#dico_str = '{"a":1, "b":2}'

dico_str=r"""
{
  "dataset": ".\\data\\Nursery\\nursery.csv",
  "node": {
    "var": "health",
    "edges": [
      {
        "edge": {
          "value": "not_recom",
          "leaf": {
            "decision": "not_recom",
            "p": 1.0
          }
        }
      },
      {
        "edge": {
          "value": "priority",
          "leaf": {
            "decision": "spec_prior",
            "p": 0.5708333333333333
          }
        }
      },
      {
        "edge": {
          "value": "recommended",
          "leaf": {
            "decision": "priority",
            "p": 0.5583333333333333
          }
        }
      },
      {
        "edge": {
          "value": "default",
          "leaf": {
            "decision": "not_recom",
            "p": 0.3333333333333333
          }
        }
      }
    ]
  }
}
"""
dico = json.loads(dico_str)
print(dico)  #this works






