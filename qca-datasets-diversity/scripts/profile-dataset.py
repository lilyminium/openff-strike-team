import functools
import pathlib

import click
import tqdm


from openff.toolkit import Molecule

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]

@functools.cache
def _smiles_to_inchi(smi: str) -> str:
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    return mol.to_inchi(fixed_hydrogens=True)


@functools.cache
def _profile(smi: str) -> int:
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    n_atoms = mol.n_atoms
    total_charge = sum([atom.formal_charge for atom in mol.atoms]).m
    n_heavy_atoms = sum(1 for atom in mol.atoms if atom.atomic_number != 1)
    total_weight = sum(atom.mass for atom in mol.atoms).m
    return {
        "n_atoms": n_atoms,
        "total_charge": total_charge,
        "n_heavy_atoms": n_heavy_atoms,
        "mw": total_weight,
    }


def _histplot(
    df: pd.DataFrame,
    column: str,
    xlabel: str,
    binrange: tuple[int, int],
    binwidth: int,
    image_directory: pathlib.Path,
):
    _, ax = plt.subplots()
    sns.histplot(
        data=df[column],
        binrange=binrange,
        binwidth=binwidth,
        ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.yscale("log")
    plt.tight_layout()

    output_file = image_directory / f"{column}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Saved histogram to {output_file}")


def _get_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

def main(
    input_tables: str = "../tables",
    image_directory: str = "../images",
):
    dataset = ds.dataset(input_tables, format="parquet")
    n_inchis = len(set(
        _smiles_to_inchi(x)
        for x in tqdm.tqdm(
            set(_get_column(dataset, "smiles")),
            desc="Converting SMILES to InChI keys"
        )
    ))

    opt_dataset = dataset.filter(pc.field("type") == "optimization")
    n_opt_inchis = len(set(
        _smiles_to_inchi(x)
        for x in tqdm.tqdm(
            set(_get_column(opt_dataset, "smiles")),
            desc="Converting SMILES to InChI keys"
        )
    ))
    
    td_dataset = dataset.filter(pc.field("type") == "torsiondrive")
    n_td_inchis = len(set(
        _smiles_to_inchi(x)
        for x in tqdm.tqdm(
            set(_get_column(td_dataset, "smiles")),
            desc="Converting SMILES to InChI keys"
        )
    ))

    print(f"Total number of unique InChI keys: {n_inchis}")
    print(f"Number of unique InChI keys in optimizations: {n_opt_inchis}")
    print(f"Number of unique InChI keys in torsiondrives: {n_td_inchis}")

    image_directory = pathlib.Path(image_directory)
    image_directory.mkdir(parents=True, exist_ok=True)

    entries = [_profile(x) for x in tqdm.tqdm(_get_column(dataset, "smiles"))]
    df = pd.DataFrame(entries)

    _histplot(
        df, 
        column="n_atoms",
        xlabel="# atoms",
        binrange=(0, round(df.n_atoms.max(), -1) + 10),
        binwidth=10,
        image_directory=image_directory,
    )
    _histplot(
        df, 
        column="n_heavy_atoms",
        xlabel="# heavy atoms",
        binrange=(0, round(df.n_heavy_atoms.max(), -1) + 10),
        binwidth=5,
        image_directory=image_directory,
    )
    _histplot(
        df, 
        column="total_charge",
        xlabel="Total charge",
        binrange=(df.total_charge.min(), df.total_charge.max()),
        binwidth=1,
        image_directory=image_directory,
    )
    _histplot(
        df,
        column="mw",
        xlabel="Molecular weight (Da)",
        binrange=(round(df.mw.min(), -1) - 10, round(df.mw.max(), -1) + 10),
        binwidth=20,
        image_directory=image_directory,
    )


if __name__ == "__main__":
    main()
