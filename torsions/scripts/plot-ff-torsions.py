from rdkit import Chem
import pathlib
import tqdm

import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np
import pandas as pd
import MDAnalysis as mda

from openff.toolkit import ForceField, Molecule, Quantity
from openff.units import unit

import click
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

def calc_torsion_energy(angle, parameter):
    angle = (angle * unit.degrees).m_as(unit.radians)
    total = 0 * unit.kilojoules_per_mole
    for k, phase, periodicity in zip(parameter.k, parameter.phase, parameter.periodicity):
        phase = phase.m_as(unit.radians)
        subtotal = k * (1 + np.cos(periodicity * angle - phase))
        total += subtotal
    return total



@click.command()
@click.option(
    "--force-field",
    "forcefield",
    type=str,
    help="The name or path of the force field to use for plotting.",
)
@click.option(
    "--output",
    "output_directory",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    help="The directory to save the plots.",
    default="../images"
)
def plot_ff_torsions(forcefield: str, output_directory: str):
    """
    Plots the torsion profiles of a force field.

    Outputs are saved in the following format:
    output_directory/forcefield_name/parameter_id/forcefield.png
    """
    name = pathlib.Path(forcefield).stem
    output_directory = pathlib.Path(output_directory) / name
    output_directory.mkdir(exist_ok=True, parents=True)
    
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    handler = ff.get_parameter_handler("ProperTorsions")
    for parameter in tqdm.tqdm(handler.parameters):
        fig, ax = plt.subplots(figsize=(4, 3))
        xs = np.linspace(-180, 180, 360)
        ys = [
            calc_torsion_energy(x, parameter).m_as(unit.kilocalories_per_mole)
            for x in xs
        ]
        ax.plot(xs, ys)
        ax.set_title(parameter.id)
        ax.set_ylabel("Energy\n[kcal/mol]")
        plt.tight_layout()
        filename = output_directory / parameter.id / f"forcefield.png"
        filename.parent.mkdir(exist_ok=True, parents=True)

        plt.savefig(filename, dpi=300)
        print(f"Saved to {filename}")
        plt.close()


if __name__ == "__main__":
    plot_ff_torsions()
