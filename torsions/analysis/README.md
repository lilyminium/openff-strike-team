

### Generate `parameter_id_to_torsion_ids.json`

```
python map-parameter-ids-to-torsions.py         \
    --input     "datasets/qm/output"            \
    --force-field   "tm-2.2.offxml"             \
    --output    parameter_id_to_torsion_ids.json
```

### Plot FF torsions

```
python plot-ff-torsions.py                      \
    --force-field       "tm-2.2.offxml"         \
    --output            ../images
```

### Plot energy breakdown
```
python plot-singlepoint-energies-breakdown.py       \
    --parameter-id t60g                             \
    --output-directory ../images                    \
    --qm-dataset    datasets/qm/output              \
    --mm-dataset    datasets/mm/singlepoint-torsiondrive-datasets   \
    --forcefield    "tm-2.2.offxml"                 \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json
```


### Plotting QM vs MM

Plotting singlepoints:

```
python plot-qm-vs-mm-profile.py                     \   
    --parameter-id t60g                             \
    --output-directory ../images/singlepoints       \
    --qm-dataset    datasets/qm/output              \
    --mm-dataset    datasets/mm/singlepoint-torsiondrive-datasets   \
    --forcefield    "tm-2.2.offxml"                 \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json

```

Plotting minimizations with RMSDs:

```
python plot-qm-vs-mm-profile.py                     \   
    --parameter-id t60g                             \
    --output-directory ../images/minimized          \
    --qm-dataset    datasets/qm/output              \
    --mm-dataset    datasets/mm/minimized-torsiondrive-datasets   \
    --with-rmsds                                    \
    --forcefield    "tm-2.2.offxml"                 \
    --parameter-ids-to-torsions parameter_id_to_torsion_ids.json

```