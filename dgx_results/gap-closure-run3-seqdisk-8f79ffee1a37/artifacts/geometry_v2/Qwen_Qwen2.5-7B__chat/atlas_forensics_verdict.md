ATLAS FORENSICS
===============
Euclidean ceiling of Eq.5/Eq.6 delta_rel: 0.2929 (analytic, unit square)
  numeric check: worst over 60k random 4-point configs = 0.2855; unit square = 0.2929
  => any Euclidean cloud satisfies delta_rel <= 0.2929; the Atlas's reported ~0.995 medians are unattainable under their own printed formula. Our measured ~0.08 is ~27% of the true ceiling.

[Qwen/Qwen2.5-7B] candidate-statistic sweep (median across prompts):
  eq56_delta_rel            : plateau=   0.061 final=   0.114
  one_minus_delta_rel       : plateau=   0.939 final=   0.886
  max_gromov_over_diam      : plateau=   0.690 final=   0.816
  defect_over_meddist       : plateau=   0.125 final=   0.203
  defect_over_mindist       : plateau=   0.645 final=   1.387
  minform_over_maxgromov    : plateau=   0.087 final=   0.141
  defect_unnorm             : plateau=   8.455 final=  67.292

