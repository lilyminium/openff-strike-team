import pathlib
import functools
import collections

import click
import tqdm

from openff.toolkit import Molecule, ForceField

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.dataset as ds

def _get_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

def single_label_row(row: dict, forcefield, empty_entry: dict[str, bool]):
    # make molecule
    smiles = row["cmiles"]
    mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)

    # set up new entry
    entry = dict(empty_entry)
    labels = forcefield.label_molecules(mol.to_topology())[0]
    dihedral_indices = [tuple(x) for x in row["dihedral_indices"]]
    for indices, parameter in labels["ProperTorsions"].items():
        indices = tuple(indices)
        if indices in dihedral_indices or indices[::-1] in dihedral_indices:
            label = parameter.id if parameter.id else parameter.name
            entry[label] = True
    return entry

@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="../tables/torsiondrive",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../labels/forcefield-torsions/openff-2.2.0",
)
@click.option(
    "--forcefield",
    "forcefield_path",
    type=str,
    default="openff-2.2.0.offxml"
)
def main(
    input_path: str,
    output_path: str,
    forcefield_path: str,
):
    forcefield = ForceField(forcefield_path)
    forcefield_name = pathlib.Path(forcefield_path).stem
    
    # Load the dataset
    dataset = ds.dataset(input_path)
    # load smiles
    all_torsiondrive_ids = set(_get_column(dataset, "torsiondrive_id"))

     # check existing output
    output_directory = pathlib.Path(output_path)
    output_directory.mkdir(exist_ok=True, parents=True)

    file_index = 0
    output_dataset = ds.dataset(output_path)
    if output_dataset.count_rows():
        # load existing output
        existing_tids = _get_column(output_dataset, "torsiondrive_id")
        # exclude existing data
        all_torsiondrive_ids = all_torsiondrive_ids - set(existing_tids)
        file_index = len(output_dataset.files)

    all_torsiondrive_ids = sorted(all_torsiondrive_ids)

    # load rows, one per torsiondrive
    seen_torsiondrive_ids = set()
    expression = pc.field("torsiondrive_id").isin(all_torsiondrive_ids)
    subset = dataset.filter(expression)
    columns = [
        "cmiles",
        "smiles",
        "torsiondrive_id", "dihedral_indices", "specification",
        "dataset"
    ]
    pylist = subset.to_table(columns=columns).to_pylist()
    all_rows = []
    for row in pylist:
        if row["torsiondrive_id"] in seen_torsiondrive_ids:
            continue
        seen_torsiondrive_ids.add(row["torsiondrive_id"])
        all_rows.append(row)
    print(f"Loaded {len(all_rows)} unique TorsionDrives to process")

    new_entries = []
    # set up empty entry
    empty_entry = {}
    handler = forcefield.get_parameter_handler("ProperTorsions")
    for parameter in handler.parameters:
        label = parameter.id if parameter.id else parameter.name
        empty_entry[label] = False
    for row in tqdm.tqdm(all_rows):
        entry = single_label_row(row, forcefield, empty_entry)
        entry["smiles"] = row["smiles"]
        entry["cmiles"] = row["cmiles"]
        entry["torsiondrive_id"] = row["torsiondrive_id"]
        entry["dataset"] = row["dataset"]
        entry["specification"] = row["specification"]
        entry["dihedral_indices"] = row["dihedral_indices"]
        entry["forcefield"] = forcefield_name
        new_entries.append(entry)
    
    table = pa.Table.from_pylist(new_entries)
    new_file = output_directory / f"batch-{file_index:04d}.parquet"
    pq.write_table(table, new_file)
        


if __name__ == "__main__":
    main()
