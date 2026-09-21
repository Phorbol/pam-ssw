"""Offline proposal outcomes from paid request boundaries and explicit ledger errors."""
def classify_material_events(data):
    valid={p['index']:p for p in data['landings'] if p['index']>=0}
    budgets=[dict(ledger_index=i,**item) for i,item in enumerate(data.get('ledger',()))
             if str(item.get('error','')).startswith('BudgetExhausted(')]
    boundary=0;rows={}
    for event in data['records']:
        start=boundary;boundary+=event['requests']
        if 'index' not in event:continue
        index=event['index']
        matched=[b for b in budgets if start<=b['before']<=boundary and b['after']<=boundary]
        if index in valid:
            outcome='valid_accepted' if valid[index]['accepted'] else 'valid_rejected'
        elif event['requests']==0 and any(b['after']<=start for b in budgets):
            outcome='not_started_no_budget'
        elif matched:
            outcome='budget_censored'
        else:outcome='failed'
        rows[index]=dict(outcome=outcome,request_interval=[start,boundary],budget_ledger_evidence=matched)
    return rows
