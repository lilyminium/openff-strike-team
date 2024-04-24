# Analysis

The first step to most of the other scripts is to
generate a mapping of parameter_id to TorsionDrive IDs
with ``python map-parameter-ids-to-torsions.py``.

After that the steps below can be largely conducted independently.

### Generate `parameter_id_to_torsion_ids.json`

```
python ../scripts/map-parameter-ids-to-torsions.py  \
    --input     "../datasets/qm/output"             \
    --force-field   "tm-2.2.offxml"                 \
    --output    parameter_id_to_torsion_ids.json
```


### Plot FF torsions

This script plots the force field torsions for each parameter.

```
python ../scripts/plot-ff-torsions.py           \
    --force-field       "tm-2.2.offxml"         \
    --output            ../images
```

An example output is `../images/tm-2.2/t60g/forcefield.png`.


### Plot all TorsionDrives for a parameter

This can be done for either singlepoint or minimized geometries.
You can optionally draw the molecules as well (`--plot-molecules`).
This can take a while so not always recommended.

```
python ../scripts/plot-all-torsiondrives.py             \
    --parameter-id      t60g                                                \
    --output-directory  ../images                                           \
    --qm-dataset        ../datasets/qm/output                               \
    --mm-dataset        ../datasets/mm/singlepoint-torsiondrive-datasets    \
    --forcefield        "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json            \
    --suffix            "-singlepoints"                                     \
    --plot-molecules
```

Produces ``../images/tm-2.2/t60g/molecules/molecules_0.png``
and ``../images/tm-2.2/t60g/mm-torsion-energies.png``.

```
python ../scripts/plot-all-torsiondrives.py             \
    --parameter-id      t60g                                                \
    --output-directory  ../images                                           \
    --qm-dataset        ../datasets/qm/output                               \
    --mm-dataset        ../datasets/mm/minimized-torsiondrive-datasets  \
    --forcefield        "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json            \
    --suffix            "-minimized"
```

Produces `../images/tm-2.2/t60g/mm-torsion-energies-minimized.png`.

### Plot energy breakdown of torsiondrives for a parameter
```
python ../scripts/plot-singlepoint-energies-breakdown.py            \
    --parameter-id      t60g                                                \
    --output-directory  ../images                                           \
    --qm-dataset        ../datasets/qm/output                               \
    --mm-dataset        ../datasets/mm/singlepoint-torsiondrive-datasets    \
    --forcefield        "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json
```


### Plotting QM vs MM

Plotting singlepoints:

```
python ../scripts/plot-qm-vs-mm-profile.py                                  \
    --parameter-id      t60g                                                \
    --output-directory  ../images                                           \
    --qm-dataset        ../datasets/qm/output                               \
    --mm-dataset        ../datasets/mm/singlepoint-torsiondrive-datasets    \
    --forcefield        "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json            \
    --suffix            "-singlepoint"

```

Produces ``../images/singlepoints/tm-2.2/t60g/17291347/qm-vs-mm-singlepoint.png``.

Plotting minimizations with RMSDs:

```
python ../scripts/plot-qm-vs-mm-profile.py                              \
    --parameter-id      t60g                                            \
    --output-directory  ../images                                       \
    --qm-dataset        ../datasets/qm/output                           \
    --mm-dataset        ../datasets/mm/minimized-torsiondrive-datasets  \
    --with-rmsds                                                        \
    --forcefield    "tm-2.2.offxml"                                     \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json        \
    --suffix        "-minimized"

```

Produces ``../images/tm-2.2/t60g/17291347/qm-vs-mm-rmsd-minimized.png``.

### Plotting individual interactions

If wanting to investigate a torsion drive in detail:

```
python ../scripts/plot-individual-energy-breakdowns.py  \
    --torsiondrive-id   17291347                                        \
    --output-directory  ../images                                       \
    --forcefield        "tm-2.2.offxml"                                 \
    --qm-dataset        ../datasets/qm/output                           \
    --mm-dataset        ../datasets/mm/minimized-torsiondrive-datasets  \
    --geometry          "mm"
```

Produces, e.g., `../images/tm-2.2/t60g/17291347/individual-energy-breakdowns-mm/images/a10_0-1-3.png` (among many others).