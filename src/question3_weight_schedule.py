"""Causal month-start weight schedules. Retrospective same-month weights are never selectable."""
from __future__ import annotations

import pandas as pd

from analyze_question3_monthly_weights import collect_samples, fit_asof
from analyze_question3_revision import calibrate

MODES=('january','january-full','previous-month','expanding')


def build_schedule(inputs,mode,legacy_report=None):
    if mode not in MODES:
        raise ValueError(f'Unknown weight mode: {mode}')
    schedule={}
    if mode=='january':
        report=calibrate(inputs) if legacy_report is None else legacy_report
        records=[{'weights':b['fit_weights'],'latest_available_at':'2025-01-31 18:00:00'}
                 for b in report['blocks']]
    else:
        samples=collect_samples(inputs)
        if mode=='january-full':
            records=fit_asof(samples,'2025-01-01','2025-02-01')
    for month in range(2,13):
        decision=pd.Timestamp(2025,month,1)
        if mode in ('previous-month','expanding'):
            start=decision-pd.offsets.MonthBegin(1) if mode=='previous-month' else pd.Timestamp('2025-01-01')
            records=fit_asof(samples,start,decision)
        else:
            start=pd.Timestamp('2025-01-04' if mode=='january' else '2025-01-01')
        if any(pd.Timestamp(r['latest_available_at'])>decision for r in records):
            raise ValueError('Weight training uses future target values')
        schedule[f'2025-{month:02d}']={'mode':mode,'effective_at':str(decision),
            'training_issue_start':str(start),'latest_training_target':max(r['latest_available_at'] for r in records),
            'blocks':records}
    return schedule
