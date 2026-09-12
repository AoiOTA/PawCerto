"""The numerical-failure schema must never become a full-set plot mean."""
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def test_scratch_plot_excludes_invalid_prefix_and_partial_cohort(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/plot_umi_results.py'
    spec=importlib.util.spec_from_file_location('umi_plot_under_test',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'OUT',tmp_path/'figures')
    rows=[dict(iteration=i,position_error=.1,orientation_error=.2,supported_feet=4.,
               final_policy_kl=.01,learning_rate=.001,pos_sigma=.01,orn_sigma=1.) for i in (1,2)]
    monkeypatch.setattr(module,'scratch_chain',lambda through:(rows,{}))
    base=tmp_path/'outputs/mujoco/umi_lab_scratch'
    for iteration in (0,4000):
        (base/f'model_{iteration}').mkdir(parents=True)
    complete=[dict(position_error_mean_m=.1,orientation_error_mean_rad=.2,inverted=False) for _ in range(16)]
    # A plausible 15-case subset and a huge invalid prefix must both stay out of
    # the full-set trend, even if every metric is finite.
    incomplete=[dict(position_error_mean_m=.05,orientation_error_mean_rad=.1,
                     inverted=False,completion_status='complete') for _ in range(15)]
    incomplete.append(dict(position_error_mean_m=1e6,orientation_error_mean_rad=3.,
                           inverted=True,completion_status='invalid',metrics_scope='pre_failure_prefix'))
    (base/'model_0/summary.json').write_text(json.dumps(dict(case_results=complete)))
    (base/'model_4000/summary.json').write_text(json.dumps(dict(case_results=incomplete)))
    (base/'paired_summary.json').write_text(json.dumps(dict(comparisons=[
        dict(iteration=0,inverted_cases=0,supported_feet_mean=4.),
        dict(iteration=4000,completion_status='incomplete',complete_cases=15,requested_cases=16,
             invalid_cases=1,complete_case_metrics=dict(position_error_mean_m=.05))])))
    figures={}
    monkeypatch.setattr(module,'save',lambda fig,name:figures.update({name:fig}))
    try:
        module.scratch_figures(4000)
        fig=figures['umi_scratch_stage1_mujoco_paired']
        fig.canvas.draw()
        for ax,expected in zip(fig.axes,(100.,.2)):
            data_lines=[line for line in ax.lines if line.get_linestyle()!=':']
            assert len(data_lines)==17  # 16 original cases plus their actual mean.
            for line in data_lines:
                np.testing.assert_array_equal(line.get_xdata(),[0])
                np.testing.assert_allclose(line.get_ydata(),[expected])
            text='\n'.join(t.get_text() for t in ax.texts)
            assert '15/16 complete' in text
            assert '1 numerical failure' in text
            assert 'No full-set mean' in text
    finally:
        for fig in figures.values():
            plt.close(fig)
