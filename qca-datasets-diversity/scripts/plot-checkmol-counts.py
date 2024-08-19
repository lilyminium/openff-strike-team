import click
import pathlib
import re

import pyarrow.dataset as ds
import pyarrow.compute as pc
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]

REGEXES = {
    "alcohols": "(?<!(hi))ol(?!ithium)",
    "thios": "[Tt]hio",
    "sulfurs": "ulf",
    "phosphorus": "hosph",
    "acyl_halides": "cyl",
    "alkyl_halides": "Alkyl ",
    "aryl_halides": "Aryl ",
    "amines": "min",
    "nitrogens": "[Nn]it",
    "azs": "[Aa]z"
}

BIG_GROUPS = [
    "Heterocycle",
    "Aromatic",

    
    "Alkane",
    "Alkene",
    "Alkyne",

    "Amine",
    "Azide",
    "Enamine",
    "Imine",
    "Hydrazine",
    "Hydrazone",

    "Alcohol",
    "Aldehyde",
    "Ether",
    "Ketone",
    "Lactone",
    "Carboxylic Acid",
    "Carboxylic Acid Ester",

    "Lactam",
    "Carboxylic Acid Amide"
]

@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default="../labels/checkmol",
)
@click.option(
    "--plot-name",
    type=str,
    default="checkmol"
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../images/functional-groups",
)
def main(
    input_path: str,
    plot_name: str,
    output_path: str,
):
    dataset = ds.dataset(input_path)
    df = dataset.to_table().to_pandas()
    columns = df.columns[:-2] # last is smiles and InChI
    counts = df[columns].sum()

    output_directory = pathlib.Path(output_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    csvfile = output_directory / f"functional-groups.csv"
    df.to_csv(csvfile)

    for group_name, regex in REGEXES.items():
        subcounts = counts[
            [bool(re.search(regex, col)) for col in counts.index]
        ]
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(subcounts)))
        ax = sns.barplot(
            x=subcounts.values,
            y=subcounts.index,
            ax=ax,
            orient="h"
        )
        ax.set_ylabel("")
        # plt.xticks(rotation=90)
        plt.tight_layout()
        img_file = output_directory / f"{plot_name}_{group_name}.png"

        plt.savefig(img_file, dpi=300)
        print(f"Saved histogram to {img_file}")

    big_groups = counts[BIG_GROUPS]
    fig, ax = plt.subplots(figsize=(5, 7))
    ax = sns.barplot(
        x=big_groups.values,
        y=big_groups.index,
        ax=ax,
        orient="h"
    )
    ax.set_ylabel("")
    plt.xscale("log")
    # plt.xticks(rotation=90)
    plt.tight_layout()
    img_file = output_directory / f"{plot_name}_big-groups.png"
    plt.savefig(img_file, dpi=300)
    print(f"Saved histogram to {img_file}")


if __name__ == "__main__":
    main()
