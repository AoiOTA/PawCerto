"""Saved-rollout metrics shared by the current UMI reports."""
import numpy as np

def metrics(a):
 v=a['metrics'];g=a['ground_supported_feet'];return dict(record_count=len(v),last_time_s=float(v[-1,0]),position_error_mean_m=float(v[:,1].mean()),position_error_rms_m=float(np.sqrt(np.mean(v[:,1]**2))),orientation_error_mean_rad=float(v[:,2].mean()),position_error_max_m=float(v[:,1].max()),minimum_root_up_dot=float(v[:,4].min()),inverted=bool((v[:,4]<0).any()),net_supported_feet_mean=float(v[:,7].mean()),ground_supported_feet_mean=float(g.mean()),zero_ground_supported_feet_fraction=float((g==0).mean()))
def aggregate(results,indices):
 if not indices:return None
 c=[results[i] for i in indices]
 return dict(case_indices=indices,position_error_mean_m=float(np.mean([x['position_error_mean_m'] for x in c])),position_error_rms_m=float(np.sqrt(np.mean([x['position_error_rms_m']**2 for x in c]))),orientation_error_mean_rad=float(np.mean([x['orientation_error_mean_rad'] for x in c])),position_error_max_m=max(x['position_error_max_m'] for x in c),minimum_root_up_dot=min(x['minimum_root_up_dot'] for x in c),inverted_cases=sum(x['inverted'] for x in c),ground_supported_feet_mean=float(np.mean([x['ground_supported_feet_mean'] for x in c])),zero_ground_supported_feet_fraction=float(np.mean([x['zero_ground_supported_feet_fraction'] for x in c])),nonfoot_contact_cases_gt_1N=sum(x['nonfoot_contact_samples_gt_1N']>0 for x in c),nonfoot_external_contact_samples_gt_1N=sum(x['nonfoot_external_contact_samples_gt_1N'] for x in c),nonfoot_self_contact_samples_gt_1N=sum(x['nonfoot_self_contact_samples_gt_1N'] for x in c))
