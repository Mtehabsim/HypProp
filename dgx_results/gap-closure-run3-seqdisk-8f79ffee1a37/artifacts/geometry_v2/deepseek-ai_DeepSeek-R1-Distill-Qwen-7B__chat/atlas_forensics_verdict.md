ATLAS FORENSICS
===============
Euclidean ceiling of Eq.5/Eq.6 delta_rel: 0.2929 (analytic, unit square)
  numeric check: worst over 60k random 4-point configs = 0.2855; unit square = 0.2929
  => any Euclidean cloud satisfies delta_rel <= 0.2929; the Atlas's reported ~0.995 medians are unattainable under their own printed formula. Our measured ~0.08 is ~27% of the true ceiling.

[deepseek-ai/DeepSeek-R1-Distill-Qwen-7B] candidate-statistic sweep (median across prompts):
  eq56_delta_rel            : plateau=   0.069 final=   0.102
  one_minus_delta_rel       : plateau=   0.931 final=   0.898
  max_gromov_over_diam      : plateau=   0.809 final=   0.760
  defect_over_meddist       : plateau=   0.107 final=   0.193
  defect_over_mindist       : plateau=   0.561 final=5225.103
  minform_over_maxgromov    : plateau=   0.089 final=   0.132
  defect_unnorm             : plateau=  19.036 final=  41.613

