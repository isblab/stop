cd high
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../scripts/sample/final_runs/high/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_|XLEDC" "CrossLinkingMassSpectrometryRestraint_Distance_|XLDSS" -pl ConnectivityRestraint_None ExcludedVolumeSphere_spc110 ExcludedVolumeSphere_gtusc ExcludedVolumeSphere_spc110_gtusc "CrossLinkingMassSpectrometryRestraint_Data_Score_XLEDC" "CrossLinkingMassSpectrometryRestraint_Data_Score_XLDSS" Total_Score -alt 0.6 0.5 -aut 1.0 1.0 -mlt 0.0 0.0 -mut 25.0 35.0
mv ../../../scripts/sample/final_runs/high/filter ./filter
cd ../low
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../scripts/sample/final_runs/low/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_|XLEDC" "CrossLinkingMassSpectrometryRestraint_Distance_|XLDSS" -pl ConnectivityRestraint_None ExcludedVolumeSphere_spc110 ExcludedVolumeSphere_gtusc ExcludedVolumeSphere_spc110_gtusc "CrossLinkingMassSpectrometryRestraint_Data_Score_XLEDC" "CrossLinkingMassSpectrometryRestraint_Data_Score_XLDSS" Total_Score -alt 0.6 0.5 -aut 1.0 1.0 -mlt 0.0 0.0 -mut 25.0 35.0
mv ../../../scripts/sample/final_runs/low/filter ./filter
cd ../optimal
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../scripts/sample/final_runs/optimal/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_|XLEDC" "CrossLinkingMassSpectrometryRestraint_Distance_|XLDSS" -pl ConnectivityRestraint_None ExcludedVolumeSphere_spc110 ExcludedVolumeSphere_gtusc ExcludedVolumeSphere_spc110_gtusc "CrossLinkingMassSpectrometryRestraint_Data_Score_XLEDC" "CrossLinkingMassSpectrometryRestraint_Data_Score_XLDSS" Total_Score -alt 0.6 0.5 -aut 1.0 1.0 -mlt 0.0 0.0 -mut 25.0 35.0
mv ../../../scripts/sample/final_runs/optimal/filter ./filter
cd ..
python plot_comparative.py > plotting_output.txt
