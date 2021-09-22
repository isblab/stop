python -u main.py param_file
# The optimized values need to be hard-coded in the modeline_manual_final_optimal script
seq 0 29 | parallel 'mkdir final_runs/high/output_{}'
seq 0 29 | parallel 'mkdir final_runs/low/output_{}'
seq 0 29 | parallel 'mkdir final_runs/optimal/output_{}'
seq 0 29 | parallel -P 30 --bar '../../../imp-custom/build/setup_environment.sh python modified_monomer_final_high.py final_runs/high/output_{} 1> final_runs/high/stdout_{}_{#}_{%}.txt 2> final_runs/high/stderr_{}_{#}_{%}.txt'
seq 0 29 | parallel -P 30 --bar '../../../imp-custom/build/setup_environment.sh python modified_monomer_final_optimal.py final_runs/optimal/output_{} 1> final_runs/optimal/stdout_{}_{#}_{%}.txt 2> final_runs/optimal/stderr_{}_{#}_{%}.txt'
seq 0 29 | parallel -P 30 --bar '../../../imp-custom/build/setup_environment.sh python modified_monomer_final_low.py final_runs/low/output_{} 1> final_runs/low/stdout_{}_{#}_{%}.txt 2> final_runs/low/stderr_{}_{#}_{%}.txt'
