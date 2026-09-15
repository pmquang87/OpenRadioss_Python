import json
r = json.load(open('tools/validation_data/coverage_results.json'))
for g in sorted(r['ranked_gaps'], key=lambda x: -x['cases_blocking'])[:15]:
    print(f"{g['family']:20} blocking={g['cases_blocking']} uses={g['cases_using']}")
