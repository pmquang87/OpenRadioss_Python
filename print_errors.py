import json
r = json.load(open('tools/validation_data/coverage_results.json'))
for c in r['cases']:
    if c['verdict'] == 'ERROR':
        print(f"{c['case_id']}: {c['verdict']} - {c.get('errors', [])}")
