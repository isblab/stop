cd high/exhaustiveness/
../../../../../imp-custom/build/setup_environment.sh python ../../../../../imp-custom/imp/modules/sampcon/pyext/src/exhaust.py -n gtusc_spc110 -p ../good_scoring_models/ -d ../../density_ranges.txt -m cpu_omp -c 10 -a -g 0.1 -gp
cd ../../low/exhaustiveness/
../../../../../imp-custom/build/setup_environment.sh python ../../../../../imp-custom/imp/modules/sampcon/pyext/src/exhaust.py -n gtusc_spc110 -p ../good_scoring_models/ -d ../../density_ranges.txt -m cpu_omp -c 10 -a -g 0.1 -gp
cd ../../optimal/exhaustiveness/
../../../../../imp-custom/build/setup_environment.sh python ../../../../../imp-custom/imp/modules/sampcon/pyext/src/exhaust.py -n gtusc_spc110 -p ../good_scoring_models/ -d ../../density_ranges.txt -m cpu_omp -c 10 -a -g 0.1 -gp
