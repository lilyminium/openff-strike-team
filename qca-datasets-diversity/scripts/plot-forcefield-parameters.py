import pathlib
import functools
import click

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from openff.toolkit import Molecule

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]

def _get_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

@functools.cache
def _smiles_to_inchi(smi: str) -> str:
    mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    return mol.to_inchi(fixed_hydrogens=True)

@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="../labels/forcefield-parameters/openff-2.2.0",
)
@click.option(
    "--original",
    "original_dataset_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="../tables",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../images/forcefield-parameters/openff-2.2.0",
)
@click.option(
    "--plot-name",
    type=str,
    default="openff-2.2.0"
)
@click.option(
    "--combinations",
    "combinations_paths",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    multiple=True,
    default=[],
)
def main(
    input_path: str,
    output_path: str,
    plot_name: str,
    original_dataset_path: str = None,
    combinations_paths: list[str] = None,
):
    dataset = ds.dataset(input_path)
    
    # filter for combinations if present
    if combinations_paths:
        if not original_dataset_path:
            raise ValueError("If `combinations` is specified, `original` must be specified")

        original = ds.dataset(original_dataset_path)

        combination_dfs = []
        for combination_file in combinations_paths:
            combination_df = pd.read_csv(combination_file)
            combination_dfs.append(combination_df)
        combination_df = pd.concat(combination_dfs)
        optimizations = combination_df[combination_df["type"] == "optimization"].id.values
        torsiondrives = combination_df[combination_df["type"] == "torsiondrive"].id.values

        expression = (
            pc.field("qcarchive_id").isin(optimizations)
            | pc.field("torsiondrive_id").isin(torsiondrives)
        )
        original = original.filter(expression)
        smiles = _get_column(original, "smiles")
        inchis = set([_smiles_to_inchi(smi) for smi in smiles])
        dataset = dataset.filter(pc.field("inchi").isin(inchis))
    
    df = dataset.to_table().to_pandas()
    skip_columns = ["smiles", "inchi", "forcefield"]
    cols = [col for col in df.columns if col not in skip_columns]
    counts = df[cols].sum()

    NAMES = {
        "Bonds": "b",
        "Angles": "a",
        "ProperTorsions": "t",
        "ImproperTorsions": "i",
    }
    
    output_directory = pathlib.Path(output_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    csvfile = output_directory / f"{plot_name}.csv"
    counts.to_csv(csvfile)

    for name, letter in NAMES.items():
        subcounts = counts[[x.startswith(letter) for x in counts.index]]
        max_bars = 100
        # chunk up subcounts if longer than max_bars
        for i in range(0, len(subcounts), max_bars):
            subcounts_chunk = subcounts[i:i + max_bars]
            _, ax = plt.subplots(figsize=(25, 4))
            ax = sns.barplot(ax=ax, data=subcounts_chunk)
            plt.yscale("log")
            plt.xticks(rotation=90)
            ax.set_title(f"{plot_name} {name}")
            plt.tight_layout()
            
            img_file = output_directory / f"{plot_name}_{name}_{i}.png"
            plt.savefig(img_file, dpi=300)
            print(f"Saved histogram to {img_file}")
        

if __name__ == "__main__":
    main()
