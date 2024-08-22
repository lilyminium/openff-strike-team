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

def _get_dataset_column(dataset, column: str):
    return dataset.to_table(columns=[column]).to_pydict()[column]

def plot_grouped_minimization_energies_singlepoint(
    torsiondrive_id: int,
    mm_dataset,
    qm_dataset,
    output_directory = "../images",
):
    subset = qm_dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    if not subset.count_rows():
        return

    if "torsiondrive_id" in mm_dataset.schema.names:
        minimized = mm_dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    else:
        minimized = mm_dataset.filter(
            pc.field("qcarchive_id").isin(_get_dataset_column(subset, "qcarchive_id"))
        )
    if not minimized.count_rows():
        return
    
    # energies can be called Torsion or energy_Torsion
    # at some point this was renamed to avoid confusion with Bond IC RMSDs etc
    if "Torsion" in minimized.schema.names:
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
    else:
        # get energy_Torsion and rename...
        geometry_df = minimized.to_table(
            columns=[
                "qcarchive_id",
                "energy_Bond",
                "energy_Angle",
                "energy_Torsion",
                "energy_vdW",
                "energy_Electrostatics",
                "energy_vdW 1-4",
                "energy_Electrostatics 1-4",
                "mm_energy"
            ]
        ).to_pandas()
        geometry_df = geometry_df.rename(
            columns={
                "energy_Bond": "Bond",
                "energy_Angle": "Angle",
                "energy_Torsion": "Torsion",
                "energy_vdW": "vdW",
                "energy_Electrostatics": "Electrostatics",
                "energy_vdW 1-4": "vdW 1-4",
                "energy_Electrostatics 1-4": "Electrostatics 1-4",
            }
        )

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
@click.option(
    "--combination",
    "combinations",
    type=str,
    help=(
        "Combinations of data sets to plot in CSV format, "
        "e.g. 'sage-2.2.0.csv' as generated in qca-datasets-report"
    ),
    multiple=True
)
def plot_all(
    parameter_id: str,
    output_directory: str = "../images",
    qm_dataset_path: str = "datasets/qm/output/torsiondrive",
    mm_dataset_path: str = "datasets/mm/singlepoint-torsiondrive-datasets",
    forcefield: str = "tm-2.2.offxml",
    parameter_ids_to_torsions_path: str = "parameter_id_to_torsion_ids.json",
    combinations: list[str] = None,
):
    qm_dataset = ds.dataset(qm_dataset_path)
    print(f"Loaded {qm_dataset.count_rows()} QM records")
    mm_dataset = ds.dataset(mm_dataset_path)
    print(f"Loaded {mm_dataset.count_rows()} MM records")
    mm_dataset = mm_dataset.filter(pc.field("forcefield") == forcefield)
    print(f"Filtered to {mm_dataset.count_rows()} MM records with forcefield {forcefield}")


    if combinations:
        allowed_torsiondrive_ids = []
        for combination in combinations:
            comb_df = pd.read_csv(combination)
            torsiondrive_ids = comb_df[comb_df.type == "torsiondrive"].id.values
            allowed_torsiondrive_ids.extend(torsiondrive_ids)

        # remove invalid
        allowed_torsiondrive_ids = set(allowed_torsiondrive_ids) - {-1}
        print(f"Found {len(allowed_torsiondrive_ids)} allowed torsiondrive ids")
        expression = pc.field("torsiondrive_id").isin(allowed_torsiondrive_ids)
        qm_dataset = qm_dataset.filter(expression)
        print(f"Filtered to {qm_dataset.count_rows()} QM records with allowed torsiondrive ids")

        if "torsiondrive_id" in mm_dataset.schema.names:
            mm_dataset = mm_dataset.filter(expression)
        else:
            expression2 = pc.field("qcarchive_id").isin(
                _get_dataset_column(qm_dataset, "qcarchive_id")
            )
            mm_dataset = mm_dataset.filter(expression2)
        print(f"Filtered to {mm_dataset.count_rows()} MM records with allowed torsiondrive ids")

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
