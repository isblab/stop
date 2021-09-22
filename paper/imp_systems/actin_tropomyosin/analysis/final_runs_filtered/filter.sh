cd high
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../modeling/final_runs/high/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_" -pl ConnectivityRestraint_None CrossLinkingMassSpectrometryRestraint_Data_Score ExcludedVolumeSphere_None GaussianEMRestraint_None SAXSRestraint_Score Total_Score -alt 1.0 -aut 1.0 -mlt 0.0 -mut 30.0
mv ../../../modeling/final_runs/high/filter ./filter
cd ../low
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../modeling/final_runs/low/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_" -pl ConnectivityRestraint_None CrossLinkingMassSpectrometryRestraint_Data_Score ExcludedVolumeSphere_None GaussianEMRestraint_None SAXSRestraint_Score Total_Score -alt 1.0 -aut 1.0 -mlt 0.0 -mut 30.0
mv ../../../modeling/final_runs/low/filter ./filter
cd ../optimal
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../modeling/final_runs/optimal/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_" -pl ConnectivityRestraint_None CrossLinkingMassSpectrometryRestraint_Data_Score ExcludedVolumeSphere_None GaussianEMRestraint_None SAXSRestraint_Score Total_Score -alt 1.0 -aut 1.0 -mlt 0.0 -mut 30.0
mv ../../../modeling/final_runs/optimal/filter ./filter
cd ../high_timed
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../modeling/final_runs_timed/high/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_" -pl ConnectivityRestraint_None CrossLinkingMassSpectrometryRestraint_Data_Score ExcludedVolumeSphere_None GaussianEMRestraint_None SAXSRestraint_Score Total_Score -alt 1.0 -aut 1.0 -mlt 0.0 -mut 30.0
mv ../../../modeling/final_runs_timed/high/filter ./filter
cd ../low_timed
../../../../imp-custom/build/setup_environment.sh python ../../../../imp-custom/imp/modules/sampcon/pyext/src/select_good.py -rd ../../../modeling/final_runs_timed/low/ -rp output_ -sl "CrossLinkingMassSpectrometryRestraint_Distance_" -pl ConnectivityRestraint_None CrossLinkingMassSpectrometryRestraint_Data_Score ExcludedVolumeSphere_None GaussianEMRestraint_None SAXSRestraint_Score Total_Score -alt 1.0 -aut 1.0 -mlt 0.0 -mut 30.0
mv ../../../modeling/final_runs_timed/low/filter ./filter
cd ..
python plot_comparative.py > plotting_output.txt
