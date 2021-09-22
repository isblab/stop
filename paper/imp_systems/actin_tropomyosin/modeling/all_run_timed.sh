seq 0 19 | parallel 'mkdir final_runs_timed/high/output_{}'
seq 0 19 | parallel 'mkdir final_runs_timed/low/output_{}'
seq 0 19 | parallel -P 20 --bar '../../imp-custom/build/setup_environment.sh python modeling_manual_final_high_timed.py final_runs_timed/high/output_{} 1> final_runs_timed/high/stdout_{}_{#}_{%}.txt 2> final_runs_timed/high/stderr_{}_{#}_{%}.txt'
seq 0 19 | parallel -P 20 --bar '../../imp-custom/build/setup_environment.sh python modeling_manual_final_low_timed.py final_runs_timed/low/output_{} 1> final_runs_timed/low/stdout_{}_{#}_{%}.txt 2> final_runs_timed/low/stderr_{}_{#}_{%}.txt'
