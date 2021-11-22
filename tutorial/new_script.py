
"""This script samples Spc110 in conjunction with gammaTuSC.
"""
from __future__ import print_function
import IMP
import RMF
import IMP.atom
import IMP.rmf
import IMP.pmi
import IMP.pmi.tools
import IMP.pmi.topology
import IMP.pmi.dof
import IMP.pmi.macros
import IMP.pmi.restraints
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.basic
import IMP.pmi.io.crosslink
import IMP.pmi.restraints.crosslinking
import ihm.cross_linkers
import os,sys

max_temp, output_path = sys.argv[1:]
max_temp = float(max_temp)

def add_gtusc_rep(mol,pdbfile,chain,unstructured_bead_size,clr):
    atomic = mol.add_structure(pdbfile,chain_id=chain,offset=0)
    mol.add_representation(atomic, resolutions=[1,20],color = clr)
    mol.add_representation(mol[:]-atomic,resolutions=[unstructured_bead_size],color=clr)
    return mol

def add_spc110_rep(mol,gtusc_pdbfile,gtusc_chain,gtusc_pdb_offset,unstructured_bead_size,clr):
    # two pdbfiles to load Spc110 from: one is the coiled coil region bound to gtusc and the other is the extra coiled coil region    # at C terminus

    gtusc_cc_structure = mol.add_structure(gtusc_pdbfile,chain_id=gtusc_chain,offset=gtusc_pdb_offset)
    # OR translate one of the chains a bit and run like below
    mol.add_representation(gtusc_cc_structure, resolutions=[1,10],color = clr)

    #central_cc_structure = mol.add_structure(central_cc_pdb_file,chain_id=central_cc_chain,offset=central_cc_offset,ca_only=True,soft_check=True)
    #mol.add_representation(central_cc_structure, resolutions=[1,10],color = clr)

    #mol.add_representation(mol[:]-gtusc_cc_structure-central_cc_structure,resolutions=[unstructured_bead_size],color = clr)

    mol.add_representation(mol[:]-gtusc_cc_structure,resolutions=[unstructured_bead_size],color = clr)

    return(mol)

###################### SYSTEM SETUP #####################
# Parameters to tweak
nframes = 5000
if '--test' in sys.argv:
    num_frames = 1000


spc110_seq_file = '../../inputs/sequence/Spc110_GS_1-220_dimer.fasta'

gtusc_seq_file = '../../inputs/sequence/5flz.fasta'

gtusc_pdbfile = '../../inputs/structure/tusc_ref14_110.pdb'


edc_file =  '../../inputs/xlinks/spc110_1_220_GCN4dimer_rjaz180_edc30mins_q0.01_psm2.txt.INPUT.txt'
dss_file =  '../../inputs/xlinks/spc110_1_220_GCN4dimer_rjaz110_dss3mins_q0.01_psm2.txt.INPUT.txt'

GTUSC_FLEX_MAX_TRANS = 5.0

SPC110_FLEX_MAX_TRANS = 4.0 # for nterm region

gtusc_missing_structure_bead_size = 20
spc110_cg_bead_size = 5

# Input sequences
gtusc_seqs = IMP.pmi.topology.Sequences(gtusc_seq_file)

spc110_seqs = IMP.pmi.topology.Sequences(spc110_seq_file)

#TODO change while changing stoichiometry
gtusc_components = {"Spc97":['A'],"Spc98":['B'],"Tub4":['C','D']}
spc110_components = {"Spc110":['E']}
gtusc_colors = {"Spc97":["light sea green"],"Spc98":["blue"],"Tub4":["goldenrod","goldenrod"]}
spc110_colors = {"Spc110":["lime green"]}

# Setup System and add a State
mdl = IMP.Model()
s = IMP.pmi.topology.System(mdl)
st = s.create_state()

# Add Molecules for each component as well as representations
mols = []
gtusc_mols = []

# first for gtusc
for prot in gtusc_components:

    for i,chain in enumerate(gtusc_components[prot]):
        if i==0:
            mol= st.create_molecule(prot,sequence=gtusc_seqs['5FLZ'+chain],chain_id=chain)
            firstmol = mol
        else:
            mol =  firstmol.create_copy(chain_id=chain)

        color = gtusc_colors[prot][i]
        mol = add_gtusc_rep(mol,gtusc_pdbfile,chain,gtusc_missing_structure_bead_size,color)
        mols.append(mol)
        gtusc_mols.append(mol)

# next for Spc110.
spc110_mols = []

for i,chain in enumerate(spc110_components['Spc110']):

    if i==0:
       mol = st.create_molecule('Spc110', sequence=spc110_seqs['Spc110'],chain_id=chain)
    else:
       mol = spc110_mols[0].create_copy(chain_id = chain)

    color = spc110_colors['Spc110'][i]

    (mol) = add_spc110_rep(mol,gtusc_pdbfile,chain,2,spc110_cg_bead_size,color)
    spc110_mols.append(mol)
    mols.append(mol)

##  calling System.build() creates all States and Molecules (and their representations)
##  Once you call build(), anything without representation is destroyed.
##  You can still use handles like molecule[a:b], molecule.get_atomic_residues() or molecule.get_non_atomic_residues()
##  However these functions will only return BUILT representations
root_hier = s.build()

# Setup degrees of freedom
#  The DOF functions automatically select all resolutions
#  Objects passed to nonrigid_parts move with the frame but also have their own independent movers.
dof = IMP.pmi.dof.DegreesOfFreedom(mdl)

# Move regions of gTuSC with unknown structure
gtusc_unstructured = []
for mol in gtusc_mols:
    gtusc_unstructured.append(mol.get_non_atomic_residues())

dof.create_flexible_beads(gtusc_unstructured,max_trans = GTUSC_FLEX_MAX_TRANS)

# DOF for Spc110
for i,mol in enumerate(spc110_mols):
   # create a rigid body for each helix

   # create floppy movers for the unstructured part
   dof.create_flexible_beads(spc110_mols[i].get_non_atomic_residues(),max_trans = SPC110_FLEX_MAX_TRANS)
   #get_non_atomic_residues is a set earlier before build, after that it returns built representations.

   dof.create_super_rigid_body(spc110_mols[i].get_non_atomic_residues(),name="spc110_NTD_srb")

#dof.create_rigid_body([spc110_mols[0][229:276],spc110_mols[1][229:276]],max_trans = 1.0 ,max_rot = 0.2,name="central_cc")

print(dof)

####################### RESTRAINTS #####################
output_objects = [] # keep a list of functions that need to be reported
display_restraints = [] # display as springs in RMF

# Connectivity keeps things connected along the backbone (ignores if inside same rigid body)
crs = []
for mol in mols:
    cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(mol,scale=4.0)
    cr.add_to_model()
    output_objects.append(cr)
    crs.append(cr)

#now add EV term
# Excluded volume - automatically more efficient due to rigid bodies
evr1 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects = spc110_mols,resolution=10)
evr1.set_label('spc110')
evr1.set_weight(1.0)
evr1.add_to_model()
output_objects.append(evr1)

evr2 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects = spc110_mols, other_objects=gtusc_mols,resolution=20)
evr2.set_weight(10.0)
evr2.set_label('spc110_gtusc')
evr2.add_to_model()
output_objects.append(evr2)

evr3 = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(included_objects = gtusc_mols,resolution=20)
evr3.set_weight(0.01)
evr3.set_label('gtusc')
evr3.add_to_model()
output_objects.append(evr3)

## Crosslink restraint
## Not using the proxl database loader for now
## spc110-gtusc edc xlinks

kw_edc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
kw_edc.set_protein1_key("PROTEIN1")
kw_edc.set_protein2_key("PROTEIN2")
kw_edc.set_residue1_key("POSITION1")
kw_edc.set_residue2_key("POSITION2")
xldb_edc = IMP.pmi.io.crosslink.CrossLinkDataBase(kw_edc)
xldb_edc.create_set_from_file(edc_file)

xlr_edc = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(root_hier=root_hier,database=xldb_edc,length=18.0,label="XLEDC",filelabel='edc',resolution=1,slope=0.03,linker=ihm.cross_linkers.edc)

xlr_edc.add_to_model()
xlr_edc.set_weight(5.0)
output_objects.append(xlr_edc)
display_restraints.append(xlr_edc)
dof.get_nuisances_from_restraint(xlr_edc) # needed to sample the nuisance particles (noise params)

#DSS xlinks
kw_dss= IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
kw_dss.set_protein1_key("PROTEIN1")
kw_dss.set_protein2_key("PROTEIN2")
kw_dss.set_residue1_key("POSITION1")
kw_dss.set_residue2_key("POSITION2")
xldb_dss = IMP.pmi.io.crosslink.CrossLinkDataBase(kw_dss)
xldb_dss.create_set_from_file(dss_file)

xlr_dss = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(root_hier=root_hier,database=xldb_dss,length=28.0,label="XLDSS",filelabel='dss',resolution=1,slope=0.03,linker=ihm.cross_linkers.dss)

xlr_dss.add_to_model()
xlr_dss.set_weight(5.0)
output_objects.append(xlr_dss)
display_restraints.append(xlr_dss)
dof.get_nuisances_from_restraint(xlr_dss) # needed to sample the nuisance particles (noise params)

####################### SAMPLING #####################
# First shuffle the system
#IMP.pmi.tools.shuffle_configuration([spc110_mols[0][0:163],spc110_mols[1][0:163],spc110_mols[0][203:276],spc110_mols[1][203:276]],max_translation=30)
IMP.pmi.tools.shuffle_configuration(spc110_mols[0][0:163],max_translation=50)
# fixing the coiled-coil that is bound to gtusc

# Quickly move all flexible beads into place
dof.optimize_flexible_beads(100)

# Run replica exchange Monte Carlo sampling
mc1=IMP.pmi.macros.ReplicaExchange0(mdl,
                                    root_hier=root_hier,                          # pass the root hierarchy
                                    crosslink_restraints=display_restraints,                     # will display like XLs
                                    monte_carlo_temperature = 1.0,
                                    replica_exchange_minimum_temperature = 1.0,
                                    replica_exchange_maximum_temperature = max_temp,
                                    num_sample_rounds = 1,
                                    monte_carlo_sample_objects=dof.get_movers(),  # pass MC movers
                                    global_output_directory=output_path,
                                    output_objects=output_objects,
                                    monte_carlo_steps=10,
                                    number_of_frames=nframes,
                                     number_of_best_scoring_models=10
				                   )

mc1.execute_macro()
