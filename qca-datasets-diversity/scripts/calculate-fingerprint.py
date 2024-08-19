import click
import collections
import functools
import pathlib
import typing
import tqdm

from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

def _get_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

@functools.cache
def _smiles_to_inchi(smi: str) -> str:
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    return mol.to_inchi(fixed_hydrogens=True)


FINGERPRINTS = {
    "morgan": rdFingerprintGenerator.GetMorganGenerator(),
    "rdkit": rdFingerprintGenerator.GetRDKitFPGenerator(),
    "atom-pair": rdFingerprintGenerator.GetAtomPairGenerator(),
    "topological-torsion": rdFingerprintGenerator.GetTopologicalTorsionGenerator(),
}


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default="../tables",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../labels/fingerprints/morgan/fingerprints",
)
@click.option(
    "--fingerprint",
    "fingerprint_type",
    type=click.Choice(["morgan", "rdkit", "atom-pair", "topological-torsion"]),
    default="morgan",
)
def compute(
    input_path: list[str],
    output_path: str,
    fingerprint_type: typing.Literal["morgan", "rdkit", "atom-pair", "topological-torsion"],
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
    
    all_smiles = sorted(set(all_smiles))
    print(f"Total number of SMILES: {len(all_smiles)}")

    fingerprinter = FINGERPRINTS[fingerprint_type]
    entries = []
    for smi in tqdm.tqdm(all_smiles):
        # convert through toolkit for explicit Hs
        offmol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
        try:
            rdmol = offmol.to_rdkit()
        except BaseException as e:
            print(e)
            continue
        fp = fingerprinter.GetFingerprint(rdmol)
        b64 = fp.ToBase64()
        entries.append({
            "smiles": smi,
            "fingerprint": b64,
            "name": fingerprint_type,
        })
    
    table = pa.Table.from_pylist(entries)
    ds.write_dataset(table, output_path, format="parquet")


if __name__ == "__main__":
    compute()
