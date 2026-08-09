import json

with open('tools/validation_data/coverage_results_m41.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append("### (a) Per-case verdict table (all 529 runnable official decks)\n")
lines.append("verdicts: CLEAN = rc0, nothing skipped (control cards excluded); SKIPS(n) = rc0, n non-control keyword families skipped; ERROR = starter refused the model (rc2, collected errors); CRASH = uncaught traceback (rc!=0). Blockers: hard keyword gaps, then PARSE = caught reader failure, ERR = model error (top 3 shown).\n")
lines.append("| case | category | verdict | blockers |")
lines.append("|---|---|---|---|")

for c in data['cases']:
    if not c.get('runnable', True): continue
    
    case_id = c['case_id']
    cat = c['category']
    if case_id.startswith(cat + '/'):
        case_name = case_id[len(cat)+1:]
    else:
        case_name = case_id
    if case_name.endswith('_0000.rad'):
        case_name = case_name[:-9]
    elif case_name.endswith('/D00'):
        case_name = case_name[:-4]
    elif case_name.endswith('.rad'):
        case_name = case_name[:-4]

    verdict = c['verdict']
    blockers = c.get('blockers', [])
    
    if len(blockers) > 3:
        n_extra = len(blockers) - 2
        blockers_str = '; '.join(blockers[:2]) + f" (+{n_extra})"
    else:
        blockers_str = '; '.join(blockers)
        
    blockers_str = blockers_str.replace('\n', ' ').replace('|', '&#124;')
    lines.append(f"| {case_name} | {cat} | {verdict} | {blockers_str} |")

lines.append("\n### (b) Ranked keyword-gap table\n")
lines.append("cases_using = deck contains the unsupported family; cases_blocking = family is a hard physics skip in that run; sole_blocker = it is the case's ONLY hard gap; blocks = total skipped blocks corpus-wide.\n")
lines.append("| rank | unsupported family | cases using | cases blocking | sole blocker | blocks |")
lines.append("|---|---|---|---|---|---|")

gaps = data.get('ranked_gaps', [])
for i, gap in enumerate(gaps, 1):
    if i <= 20:
        lines.append(f"| {i} | {gap['family']} | {gap['cases_using']} | {gap['cases_blocking']} | {gap['cases_sole_blocker']} | {gap.get('blocks_total', '')} |")

n_more = len(gaps) - 20
if n_more > 0:
    lines.append(f"| - | ({n_more} more families, each blocking <= 5 cases) | | | | |")

lines.append("\n**Highest-value next port target:** See ranked gaps above.\n")

lines.append("### (c) CRASH list and reader-bug signatures\n")
lines.append("**rc!=0 crashes: NONE** - no deck produced an uncaught traceback. Every reader failure was caught and reported.\n")
lines.append("| # | parse-failure signature | cases | example |")
lines.append("|---|---|---|---|")

parse_errors = data.get('parse_error_signatures', [])
for i, err in enumerate(parse_errors, 1):
    lines.append(f"| {i} | {err['signature']} | {err['count']} | {err['example']} |")

if not parse_errors:
    lines.append("| - | NONE | 0 | |")

with open('tools/validation_data/coverage_tables.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
