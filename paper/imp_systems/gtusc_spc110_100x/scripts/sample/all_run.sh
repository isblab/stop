seq 0 29 | parallel 'mkdir final_runs/high/output_{}'
seq 0 29 | parallel 'mkdir final_runs/low/output_{}'
seq 0 29 | parallel -P 30 --bar '../../../imp-custom/build/setup_environment.sh python modified_monomer_final_high.py final_runs/high/output_{} 1> final_runs/high/stdout_{}_{#}_{%}.txt 2> final_runs/high/stderr_{}_{#}_{%}.txt'
seq 0 29 | parallel -P 30 --bar '../../../imp-custom/build/setup_environment.sh python modified_monomer_final_low.py final_runs/low/output_{} 1> final_runs/low/stdout_{}_{#}_{%}.txt 2> final_runs/low/stderr_{}_{#}_{%}.txt'
