import json

with open('coverage.json') as f:
    j = json.loads(f.read())
print(round(float(j['totals']['percent_covered']), 2))