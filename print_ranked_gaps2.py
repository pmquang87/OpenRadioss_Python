import json
r = json.load(open('tools/validation_data/coverage_results.json'))
for g in sorted(r['ranked_gaps'], key=lambda x: -x.get('blocks_total', 0))[:20]:
    print(f"{g['family']:20} blocks_total={g.get('blocks_total', 0)} uses={g['cases_using']}")
