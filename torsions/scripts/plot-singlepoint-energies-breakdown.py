from rdkit import Chem
import pathlib
import itertools
import tempfile
import tqdm
import os
import json

import click
import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np
import pandas as pd
import MDAnalysis as mda

from openff.toolkit import ForceField, Molecule, Quantity
from openff.units import unit

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]

OPENFF_BLUE = "#015480"
OPENFF_LIGHT_BLUE = "#2F9ED2"
OPENFF_ORANGE = "#F08521"
OPENFF_RED = "#F03A21"
OPENFF_GRAY = "#3E424A"

COLORS = {
    "blue": OPENFF_BLUE,
    "cyan": OPENFF_LIGHT_BLUE,
    "orange": OPENFF_ORANGE,
    "red": OPENFF_RED,
    "gray": OPENFF_GRAY
}

def plot_grouped_minimization_energies_singlepoint(
    torsiondrive_id: int,
    mm_dataset,
    qm_dataset,
    output_directory = "../images",
):
    subset = qm_dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)

    minimized = mm_dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    geometry_df = minimized.to_table(
        columns=[
            "qcarchive_id",
            "Bond",
            "Angle",
            "Torsion",
            "vdW",
            "Electrostatics",
            "vdW 1-4",
            "Electrostatics 1-4",
            "mm_energy"
        ]
    ).to_pandas()
    qca_ids_df = subset.to_table().to_pandas()
    df = qca_ids_df.merge(geometry_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="inner")
    df["Total"] = df["mm_energy"]
    df = df.sort_values("grid_id")
    
    melted = df.melt(
        id_vars=["grid_id", "qcarchive_id"],
        value_vars=[
            "Bond",
            "Angle",
            "Torsion",
            "vdW",
            "Electrostatics",
            "vdW 1-4",
            "Electrostatics 1-4",
            "Total"
        ],
        value_name="Energy [kcal/mol]",
        var_name="Type",
    )

    g = sns.FacetGrid(
        data=melted,
        hue="Type",
        aspect=1.5,
        height=3.5,
    )
    g.map(sns.lineplot, "grid_id", "Energy [kcal/mol]")
    ax = list(g.axes.flatten())[0]
    ax.set_title(f"TorsionDrive {torsiondrive_id}")
    ax.set_xlabel("Angle (°)")
    g.add_legend()

    output_directory.mkdir(exist_ok=True, parents=True)
    file = output_directory / "singlepoint-energies-breakdown.png"
    g.savefig(file, dpi=300)
    print(f"Saved to {file}")


@click.command()
@click.option(
    "--parameter-id",
    type=str,
    help="The parameter id to plot.",
)
@click.option(
    "--output-directory",
    type=str,
    help="The directory to save the plots.",
    default="../images"
)
@click.option(
    "--qm-dataset",
    "qm_dataset_path",
    type=str,
    help="The path to the QM dataset.",
    default="datasets/qm/output/torsiondrive"
)
@click.option(
    "--mm-dataset",
    "mm_dataset_path",
    type=str,
    help="The path to the MM dataset.",
    default="datasets/mm/singlepoint-torsiondrive-datasets"
)
@click.option(
    "--forcefield",
    type=str,
    help="The forcefield to use.",
    default="tm-2.2.offxml"
)
@click.option(
    "--parameter-ids-to-torsions",
    "parameter_ids_to_torsions_path",
    type=str,
    help="The path to the parameter id to torsion ids mapping.",
    default="parameter_id_to_torsion_ids.json"
)
def plot_all(
    parameter_id: str,
    output_directory: str = "../images",
    qm_dataset_path: str = "datasets/qm/output/torsiondrive",
    mm_dataset_path: str = "datasets/mm/singlepoint-torsiondrive-datasets",
    forcefield: str = "tm-2.2.offxml",
    parameter_ids_to_torsions_path: str = "parameter_id_to_torsion_ids.json",
):
    qm_dataset = ds.dataset(qm_dataset_path)
    mm_dataset = ds.dataset(mm_dataset_path)
    mm_dataset = mm_dataset.filter(pc.field("forcefield") == forcefield)

    with open(parameter_ids_to_torsions_path, "r") as f:
        parameter_id_to_torsion_ids = json.load(f)

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    ff_name = pathlib.Path(forcefield).stem

    torsion_ids = parameter_id_to_torsion_ids[parameter_id]
    for torsion_id in tqdm.tqdm(torsion_ids):
        plot_grouped_minimization_energies_singlepoint(
            torsion_id,
            mm_dataset,
            qm_dataset,
            output_directory=(
                output_directory
                / ff_name
                / parameter_id
                / str(torsion_id)
            )
        )


if __name__ == "__main__":
    plot_all()
