../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_high.py timing/output_0 1> timing/stdout_0.txt 2> timing/stderr_0.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_high.py timing/output_1 1> timing/stdout_1.txt 2> timing/stderr_1.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_high.py timing/output_2 1> timing/stdout_2.txt 2> timing/stderr_2.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_low.py timing/output_3 1> timing/stdout_3.txt 2> timing/stderr_3.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_low.py timing/output_4 1> timing/stdout_4.txt 2> timing/stderr_4.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_low.py timing/output_5 1> timing/stdout_5.txt 2> timing/stderr_5.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_optimal.py timing/output_6 1> timing/stdout_6.txt 2> timing/stderr_6.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_optimal.py timing/output_7 1> timing/stdout_7.txt 2> timing/stderr_7.txt &
../../../imp-custom/build/setup_environment.sh python -u modified_monomer_final_optimal.py timing/output_8 1> timing/stdout_8.txt 2> timing/stderr_8.txt &
python timing_script.py
