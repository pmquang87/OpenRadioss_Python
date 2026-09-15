import json
r = json.load(open('tools/validation_data/coverage_results.json'))
for c in r['cases']:
    for g in c.get('gaps', []):
        if any(x in g for x in ('BOX', 'INTER', 'FAIL')):
            print(f"{c['case_id']}: {g}")
