from __future__ import print_function
import IMP
import IMP.pmi
import IMP.pmi.io
import IMP.pmi.io.crosslink
import IMP.pmi.topology
import IMP.pmi.macros
import IMP.pmi.restraints
import IMP.pmi.restraints.stereochemistry
import IMP.pmi.restraints.saxs
import IMP.pmi.restraints.crosslinking
import IMP.pmi.restraints.em
import IMP.pmi.dof
import ihm.cross_linkers
import IMP.atom
import IMP.saxs
import sys

# GET THE PARAMETERS FROM THE COMMAND LINE
output_path_from_cl = sys.argv[1]

tropo_rb_mtrans = 0.4
tropo_rb_mrot = 0.1
actgel_rb_mtrans = 0.1
actgel_rb_mrot = 0.1
bead_mt = 4
max_temp = 6

pdb_dir = "../data/pdb/"
fasta_dir = "../data/fasta/"
saxs_data = "./derived_data/saxs/4pki.pdb.0.15.dat"
xl_data = "./derived_data/xl/derived_xls.dat"
gmm_data = "./derived_data/em/4pki_20a_50.gmm"

sequences = IMP.pmi.topology.Sequences(fasta_dir + "4pkh.fasta.txt")

xl_weight = 10.0
em_weight = 10.0
saxs_weight = 10.0

mdl = IMP.Model()

sys = IMP.pmi.topology.System(mdl)

st = sys.create_state()

actin = st.create_molecule("actin", sequence=sequences["actin"])
geltrop = st.create_molecule("geltrop", sequence=sequences["gelsolin-tropomyosin"])

a1 = actin.add_structure(
    pdb_dir + "4pki.pdb",
    chain_id='A')
a21 = geltrop.add_structure(
    pdb_dir + "4pki.pdb",
    chain_id='G',
    res_range=(52, 177),
    offset=-51)
a22 = geltrop.add_structure(
    pdb_dir + "4pki.pdb",
    chain_id='G',
    res_range=(1170, 1349),
    offset=-1170 - 51 + 196)

actin.add_representation(
    a1,
    resolutions=[1, 10],
    density_residues_per_component=10,
    density_prefix="./gmm_files/actin_gmm",
    density_force_compute=False,
    density_voxel_size=3.0)

actin.add_representation(
    actin[:] - a1,
    resolutions=[1],
    setup_particles_as_densities=True)

geltrop.add_representation(
    geltrop.get_atomic_residues() and geltrop[:126], resolutions=[1, 10],
    density_residues_per_component=10,
    density_prefix="./gmm_files/gelsolin_gmm",
    density_force_compute=False,
    density_voxel_size=3.0)

geltrop.add_representation(
    geltrop.get_atomic_residues() and geltrop[144:], resolutions=[1, 10],
    density_residues_per_component=10,
    density_prefix="./gmm_files/tropomyosin_gmm",
    density_force_compute=False,
    density_voxel_size=3.0)

geltrop.add_representation(
    geltrop.get_non_atomic_residues(),
    resolutions=[1],
    setup_particles_as_densities=True)

root_hier = sys.build()

dof = IMP.pmi.dof.DegreesOfFreedom(mdl)

tropo_rb_resis = geltrop[144:]
rb1 = dof.create_rigid_body(
    tropo_rb_resis,
    max_trans=tropo_rb_mtrans,
    max_rot=tropo_rb_mrot,
    nonrigid_parts=tropo_rb_resis & geltrop.get_non_atomic_residues(),
    nonrigid_max_trans=bead_mt)
print('Tropo RB:', rb1)
actgel_rb_resis = geltrop[:126]
actgel_rb_resis |= actin[:]

nars = geltrop[:126] & geltrop.get_non_atomic_residues()
nars |= actin.get_non_atomic_residues()

rb2 = dof.create_rigid_body(
    actgel_rb_resis, max_trans=actgel_rb_mtrans, max_rot=actgel_rb_mrot, nonrigid_parts=nars,
    nonrigid_max_trans=bead_mt)
print('ActGel RB:', rb2)
fb_movers = dof.create_flexible_beads(
    geltrop.get_non_atomic_residues(), max_trans=bead_mt)

output_objects = []

cr = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(actin)
cr.add_to_model()
output_objects.append(cr)

cr2 = IMP.pmi.restraints.stereochemistry.ConnectivityRestraint(geltrop)
cr2.add_to_model()
output_objects.append(cr2)

evr = IMP.pmi.restraints.stereochemistry.ExcludedVolumeSphere(
    included_objects=[actin, geltrop],
    resolution=1000)
output_objects.append(evr)

sr = IMP.pmi.restraints.saxs.SAXSRestraint(
    input_objects=[actin, geltrop],
    saxs_datafile=saxs_data,
    weight=saxs_weight,
    ff_type=IMP.saxs.RESIDUES,
    maxq=0.15)
output_objects.append(sr)

xldbkc = IMP.pmi.io.crosslink.CrossLinkDataBaseKeywordsConverter()
xldbkc.set_standard_keys()

xldb = IMP.pmi.io.crosslink.CrossLinkDataBase()
xldb.create_set_from_file(file_name=xl_data,
                          converter=xldbkc)

xlr = IMP.pmi.restraints.crosslinking.CrossLinkingMassSpectrometryRestraint(
    root_hier=root_hier,
    database=xldb,
    length=25,
    resolution=1,
    slope=0.000001,
    weight=xl_weight,
    linker=ihm.cross_linkers.dss)

xlr.add_to_model()
output_objects.append(xlr)

densities = IMP.atom.Selection(
    root_hier, representation_type=IMP.atom.DENSITIES).get_selected_particles()

emr = IMP.pmi.restraints.em.GaussianEMRestraint(
    densities,
    target_fn=gmm_data,
    slope=0.00000001,
    scale_target_to_mass=True,
    weight=em_weight)
emr.add_to_model()
output_objects.append(emr)

IMP.pmi.tools.shuffle_configuration(root_hier,
                                    max_translation=50)

dof.optimize_flexible_beads(500)

evr.add_to_model()
emr.add_to_model()
sr.add_to_model()

rex = IMP.pmi.macros.ReplicaExchange0(
    mdl,
    root_hier=root_hier,
    crosslink_restraints=[xlr],
    monte_carlo_sample_objects=dof.get_movers(),
    global_output_directory=output_path_from_cl,
    output_objects=output_objects,
    monte_carlo_steps=10,
    number_of_best_scoring_models=0,
    number_of_frames=10000,
    replica_exchange_maximum_temperature=max_temp,
    replica_exchange_minimum_temperature=1)

rex.execute_macro()
