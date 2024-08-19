import click
import collections
import functools
import pathlib
import tqdm

from openff.toolkit import Molecule
from yammbs.checkmol import ChemicalEnvironment, analyze_functional_groups

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

def _get_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

@functools.cache
def _smiles_to_inchi(smi: str) -> str:
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    return mol.to_inchi(fixed_hydrogens=True)


def single_label_smiles(smiles: str, empty_entry: dict[str, bool]):
    entry = dict(empty_entry)
    groups = analyze_functional_groups(smiles)
    if groups:
        for group in groups:
            entry[group.value] = True
    return entry


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="../tables",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../labels/checkmol",
)
def main(
    input_path: str,
    output_path: str,
):
    
    # Load the dataset
    dataset = ds.dataset(input_path)
    # load smiles
    all_smiles = _get_column(dataset, "smiles")
    all_inchis = collections.defaultdict(list)
    for smi in tqdm.tqdm(all_smiles):
        inchi = _smiles_to_inchi(smi)
        all_inchis[inchi].append(smi)
    # just take the first one for efficiency
    all_smiles = [smiles[0] for smiles in all_inchis.values()]
    smiles_to_inchi = {
        smiles[0]: inchi
        for inchi, smiles in all_inchis.items()
    }

    # check existing output
    output_directory = pathlib.Path(output_path)
    output_directory.mkdir(exist_ok=True, parents=True)

    file_index = 0
    output_dataset = ds.dataset(output_path)
    if output_dataset.count_rows():
        # load existing output
        existing_inchi = _get_column(output_dataset, "inchi")
        # exclude existing data
        remaining_inchi = set(all_inchis.keys()) - set(existing_inchi)
        all_smiles = [
            all_inchis[inchi][0]
            for inchi in remaining_inchi
        ]
        file_index = len(output_dataset.files)


    print(f"Loaded {len(all_smiles)} smiles to process")
    new_entries = []
    columns = sorted([
        col.value for col in ChemicalEnvironment
    ])
    empty_entry = dict.fromkeys(columns, False)
    for smi in tqdm.tqdm(all_smiles):
        entry = single_label_smiles(smi, empty_entry)
        entry["smiles"] = smi
        entry["inchi"] = smiles_to_inchi[smi]
        new_entries.append(entry)
    
    table = pa.Table.from_pylist(new_entries)
    new_file = output_directory / f"batch-{file_index:04d}.parquet"
    pq.write_table(table, new_file)
        


if __name__ == "__main__":
    main()
