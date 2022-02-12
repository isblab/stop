import json
import sys

with open('coverage.json') as f:
    j = json.loads(f.read())

val = round(float(j['totals']['percent_covered']), 2)

if val >= 85:
    col = 'green'
elif val >=50:
    col = 'yellow'
elif val >= 30:
    col = 'orange'
else:
    col = 'red'

if '--color' in sys.argv:
    print(col)
else:
    print(val)