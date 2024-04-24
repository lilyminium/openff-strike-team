import itertools
import typing
import os
import pathlib
import tempfile

import click
from openff.toolkit import Molecule, ForceField
from openff.units import unit

import numpy as np
import pandas as pd
import MDAnalysis as mda
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl


sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]


def draw_single_indices(mol, indices, width=300, height=300):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import rdMolDraw2D
    from matplotlib import pyplot as plt
    from cairosvg import svg2png
    

    rdmol = mol.to_rdkit()
    indices = list(map(int, indices))
    for index in indices:
        atom = rdmol.GetAtomWithIdx(int(index))
        atom.SetProp("atomNote", str(index))
    indices_text = "-".join(list(map(str, indices)))
    
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    # options.baseFontSize = 1
    drawer.DrawMolecule(rdmol, highlightAtoms=tuple(indices))
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        svg2png(bytestring=svg, write_to='tmp.png', scale=10)
        png = plt.imread("tmp.png")
        os.chdir(cwd)
    
    return png


def plot_minimization_energies(
    df,
    mol,
    atom_indices: tuple[int, ...] = (4, 2, 7, 0),
    parameter_type: str = "ImproperTorsions",
    output_directory = "single-molecules",
):
    subdf = pd.DataFrame(df[
        (df.parameter_type == parameter_type)
        & (df.atom_indices == atom_indices)
    ])

    # for torsions
    if sum(subdf.value < -150) and sum(subdf.value > 150):
        # convert all values to positive
        vals = []
        for val in subdf["value"].values:
            if val < 0:
                val += 360
            vals.append(val)
        subdf["value"] = vals
    
    color1 = "tab:blue"
    color2 = "tab:red"
    
    fig, (ax1, imgax) = plt.subplots(figsize=(12, 5), ncols=2)
    ax1.set_xlabel("Angle (°)")

    YLABELS = {
        "Bonds": "Distance (Å)",
        "Angles": "Angle (°)",
        "ProperTorsions": "Angle (°)",
        "ImproperTorsions": "Angle (°)",
        "Electrostatics": "Distance (Å)",
        "vdW": "Distance (Å)",
        "Electrostatics 1-4": "Distance (Å)",
        "vdW 1-4": "Distance (Å)",
    }
    ylabel = YLABELS[parameter_type]

    ax1.set_ylabel(ylabel, color=color1)
    ax1.plot(subdf["grid_id"], subdf["value"], color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Energy [kcal/mol]", color=color2)
    ax2.plot(subdf["grid_id"], subdf["energy"], color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ix = "-".join(map(str, atom_indices))
    parameter_id = subdf.parameter_id.values[0]
    parameter_smirks = subdf.parameter_smirks.values[0]

    if parameter_id:
        ax2.set_title(f"{parameter_type}: {ix} ({parameter_id})\n{parameter_smirks}")
    else:
        elements = subdf.elements.values[0]
        ax2.set_title(f"{parameter_type}: {ix} ({'-'.join(elements)})")

    png = draw_single_indices(mol, atom_indices)
    imgax.imshow(png, rasterized=True)
    imgax.set_xticks([])
    imgax.set_yticks([])
    imgax.spines["left"].set_visible(False)
    imgax.spines["right"].set_visible(False)
    imgax.spines["top"].set_visible(False)
    imgax.spines["bottom"].set_visible(False)
    
    plt.tight_layout()

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    if parameter_id:
        filename = f"{parameter_id}_{ix}.png"
    else:
        parameter_type = subdf.parameter_type.values[0]
        if " " in parameter_type:
            suffix = parameter_type.split()[-1]
        else:
            suffix = ""
        filename = f"{parameter_type[0]}{suffix}_{ix}.png"
    file = output_directory / filename
    plt.savefig(file, dpi=300)
    print(f"Saved to {file}")
    plt.close()





def calc_bond_energy(length, parameter):
    length = length * unit.angstrom
    return (parameter.k / 2) * ((length - parameter.length) ** 2)


def calc_angle_energy(angle, parameter):
    angle = angle * unit.degrees
    return (parameter.k / 2) * ((angle - parameter.angle) ** 2)


def calc_torsion_energy(angle, parameter):
    angle = (angle * unit.degrees).m_as(unit.radians)
    total = 0 * unit.kilojoules_per_mole
    for k, phase, periodicity in zip(parameter.k, parameter.phase, parameter.periodicity):
        phase = phase.m_as(unit.radians)
        subtotal = k * (1 + np.cos(periodicity * angle - phase))
        total += subtotal
    return total


def calc_els(distance, q1, q2, scaling_factor=1):
    distance = distance * unit.angstrom
    if distance > 9 * unit.angstrom:
        return 0 * unit.kilojoules_per_mole

    q1 = q1 * unit.elementary_charge
    q2 = q2 * unit.elementary_charge
    term = (q1 * q2) / distance

    coefficient = 1 / (4 * np.pi * unit.epsilon_0)
    return scaling_factor * coefficient * term * unit.avogadro_constant


def calc_ljs(distance, vdw1, vdw2, scaling_factor=1):
    distance = distance * unit.angstrom
    if distance > 9 * unit.angstrom:
        return 0 * unit.kilojoules_per_mole

    sigma = (vdw1.sigma + vdw2.sigma) / 2
    epsilon = (vdw1.epsilon * vdw2.epsilon) ** 0.5
    term = 4 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)

    return scaling_factor * term


def calculate_energy_breakdown(mol, labels):
    u = mda.Universe(mol.to_rdkit())
    charges = [x.m for x in mol.partial_charges]
    all_entries = []

    # valence
    FUNCTIONS = {
        "Bonds": (calc_bond_energy, lambda x: x.bond.value()),
        "Angles": (calc_angle_energy, lambda x: x.angle.value()),
        "ProperTorsions": (calc_torsion_energy, lambda x: x.dihedral.value()),
    }
    for parameter_type, functions in FUNCTIONS.items():
        energy_calculator, value_calculator = functions
        for indices, parameter in labels[parameter_type].items():
            indices = list(indices)
            value = value_calculator(u.atoms[indices])
            energy = energy_calculator(value, parameter)
            entry = {
                "atom_1": -1,
                "atom_2": -1,
                "atom_3": -1,
                "atom_4": -1,
                "atom_indices": tuple(indices),
                "element_1": "",
                "element_2": "",
                "element_3": "",
                "element_4": "",
                "elements": tuple(u.atoms[indices].elements.tolist()),
                "parameter_type": parameter_type,
                "parameter_id": parameter.id,
                "parameter_smirks": parameter.smirks,
                "value": value,
                "energy": energy.m_as(unit.kilocalories_per_mole),
            }
            for i, index in enumerate(indices, 1):
                entry[f"atom_{i}"] = index
                entry[f"element_{i}"] = u.atoms[index].element
            all_entries.append(entry)
        
    parameter_type = "ImproperTorsions"
    for key, parameter in labels["ImproperTorsions"].items():
        key = np.array(list(key))
        non_central_indices = [key[0], key[2], key[3]]
        for permuted_key in [
            (
                non_central_indices[i],
                non_central_indices[j],
                non_central_indices[k],
            )
            for (i, j, k) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        ]:
            combination = np.array([key[1], *permuted_key])
            value = u.atoms[combination].improper.value()
            energy = (calc_torsion_energy(value, parameter) / 3)
            reordered = [permuted_key[0], key[1], permuted_key[1], permuted_key[2]]
            entry = {
                "atom_1": reordered[0],
                "atom_2": reordered[1],
                "atom_3": reordered[2],
                "atom_4": reordered[3],
                "atom_indices": tuple(reordered),
                "element_1": u.atoms[reordered[0]].element,
                "element_2": u.atoms[reordered[1]].element,
                "element_3": u.atoms[reordered[2]].element,
                "element_4": u.atoms[reordered[3]].element,
                "elements": tuple(u.atoms[reordered].elements.tolist()),
                "parameter_type": parameter_type,
                "parameter_id": parameter.id,
                "parameter_smirks": parameter.smirks,
                "value": value,
                "energy": energy.m_as(unit.kilocalories_per_mole),
            }
            all_entries.append(entry)
        
    # electrostatics
    # 1-4s
    indices_14s = [(key[0], key[-1]) for key in labels["ProperTorsions"].keys()]
    distance_calculator = FUNCTIONS["Bonds"][1]
    for i, j in indices_14s:
        q1 = charges[i]
        q2 = charges[j]
        distance = distance_calculator(u.atoms[[i, j]])
        energy = calc_els(distance, q1, q2, scaling_factor=0.8333333333)
        entry = {
            "atom_1": i,
            "atom_2": j,
            "atom_3": -1,
            "atom_4": -1,
            "atom_indices": (i, j),
            "element_1": u.atoms[i].element,
            "element_2": u.atoms[j].element,
            "element_3": "",
            "element_4": "",
            "elements": tuple(u.atoms[[i, j]].elements.tolist()),
            "parameter_type": "Electrostatics 1-4",
            "parameter_id": "",
            "parameter_smirks": "",
            "value": distance,
            "energy": energy.m_as(unit.kilocalories_per_mole),
        }
        all_entries.append(entry)

        vdw1 = labels["vdW"][(i,)]
        vdw2 = labels["vdW"][(j,)]
        energy = calc_ljs(distance, vdw1, vdw2, scaling_factor=0.5)
        entry = {
            "atom_1": i,
            "atom_2": j,
            "atom_3": -1,
            "atom_4": -1,
            "atom_indices": (i, j),
            "element_1": u.atoms[i].element,
            "element_2": u.atoms[j].element,
            "element_3": "",
            "element_4": "",
            "elements": tuple(u.atoms[[i, j]].elements.tolist()),
            "parameter_type": "vdW 1-4",
            "parameter_id": "",
            "parameter_smirks": "",
            "value": distance,
            "energy": energy.m_as(unit.kilocalories_per_mole),
        }
        all_entries.append(entry)

    all_combinations = itertools.combinations(range(len(u.atoms)), 2)
    seen = set()
    for i, j in labels["Bonds"]:
        seen.add((i, j))
    for i, _, j in labels["Angles"]:
        seen.add((i, j))
    for i, j in all_combinations:
        if (i, j) in indices_14s or (j, i) in indices_14s:
            continue
        if (i, j) in seen or (j, i) in seen:
            continue
        
        q1 = charges[i]
        q2 = charges[j]
        distance = distance_calculator(u.atoms[[i, j]])
        energy = calc_els(distance, q1, q2)
        entry = {
            "atom_1": i,
            "atom_2": j,
            "atom_3": -1,
            "atom_4": -1,
            "atom_indices": (i, j),
            "element_1": u.atoms[i].element,
            "element_2": u.atoms[j].element,
            "element_3": "",
            "element_4": "",
            "elements": tuple(u.atoms[[i, j]].elements.tolist()),
            "parameter_type": "Electrostatics",
            "parameter_id": "",
            "parameter_smirks": "",
            "value": distance,
            "energy": energy.m_as(unit.kilocalories_per_mole),
        }
        all_entries.append(entry)

        vdw1 = labels["vdW"][(i,)]
        vdw2 = labels["vdW"][(j,)]
        energy = calc_ljs(distance, vdw1, vdw2)
        entry = {
            "atom_1": i,
            "atom_2": j,
            "atom_3": -1,
            "atom_4": -1,
            "atom_indices": (i, j),
            "element_1": u.atoms[i].element,
            "element_2": u.atoms[j].element,
            "element_3": "",
            "element_4": "",
            "elements": tuple(u.atoms[[i, j]].elements.tolist()),
            "parameter_type": "vdW",
            "parameter_id": "",
            "parameter_smirks": "",
            "value": distance,
            "energy": energy.m_as(unit.kilocalories_per_mole),
        }
        all_entries.append(entry)

    df = pd.DataFrame(all_entries)
    return df


@click.command()
@click.option(
    "--forcefield",
    type=str,
    help="The forcefield to use.",
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
    default="datasets/qm/output/torsiondrive",
)
@click.option(
    "--mm-dataset",
    "mm_dataset_path",
    type=str,
    help="The path to the MM dataset.",
    default="datasets/mm/singlepoint-torsiondrive-datasets",
    required=False
)
@click.option(
    "--geometry",
    type=click.Choice(["qm", "mm"]),
    help="The geometry to use. `qm` for single points, `mm` for minimized geometries.",
    default="qm"
)
@click.option(
    "--torsiondrive-id",
    type=int,
    help="The torsiondrive id to use.",
    required=True
)
def main(
    forcefield: str,
    qm_dataset_path: str,
    torsiondrive_id: int,
    output_directory: str,
    mm_dataset_path: str = None,
    geometry: typing.Literal["qm", "mm"] = "qm"
):
    """
    Calculate the contribution of every single interaction
    to the entire energy profile.

    The sum of the interactions is not perfectly equal to the
    actual total MM energy; instead of using OpenMM,
    interactions are manually calculated. In particular,
    when comparing the individually calculated contributions to the
    energy breakdown in `plot-singlepoint-energies-breakdown.py`,
    most of the time the sum of the interactions is within 1e-3
    of the parameter type. However, I have noticed some differences
    in the Electrostatics 1-4 breakdowns that I have not been
    able to isolate.

    Outputs are saved in the following structure:
    ```
    output_directory
    ├── forcefield
    │   ├── parameter_id
    │   │   ├── torsiondrive_id
    │   │   │   ├── individual-energy-breakdowns-{geometry}
    │   │   │   │   ├── individual-energy-breakdowns.csv
    │   │   │   │   ├── images
    │   │   │   │   │   ├── {parameter_id}_{atom_indices}.png
    ```
    """
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    from openff.toolkit.topology import Molecule
    from openff.units import unit

    import tqdm

    qm_dataset = ds.dataset(qm_dataset_path)

    if mm_dataset_path is not None:
        mm_dataset = ds.dataset(mm_dataset_path)
        if "forcefield" in mm_dataset.schema.names:
            mm_dataset = mm_dataset.filter(pc.field("forcefield") == forcefield)
    
    if geometry == "qm":
        coordinate_column = "conformer"
    elif geometry == "mm":
        assert mm_dataset_path is not None, "Must provide an MM dataset path."
        coordinate_column = "mm_coordinates"
    else:
        raise ValueError(f"Invalid geometry '{geometry}'.")

    expression = pc.field("torsiondrive_id") == torsiondrive_id
    qm_dataset = qm_dataset.filter(expression)
    columns = ["qcarchive_id", "grid_id", "mapped_smiles", "dihedral"]
    if geometry == "qm":
        columns.append("conformer")
    df = qm_dataset.to_table(columns=columns).to_pandas()

    if geometry == "mm":
        mm_df = mm_dataset.to_table(
            columns=["qcarchive_id", "mm_coordinates"]
        )
        mm_df = mm_df.to_pandas()
        df = df.merge(mm_df, on="qcarchive_id")
    
    df = df.sort_values("grid_id")

    # assign charges overall to ensure electrostatics are consistent
    mol = Molecule.from_mapped_smiles(
        df.mapped_smiles.values[0],
        allow_undefined_stereo=True,
    )
    mol.assign_partial_charges("am1bccelf10")

    # assign labels for all parameters to calculate
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    labels = ff.label_molecules(mol.to_topology())[0]

    # get parameter corresponding to dihedral rotation
    dihedral = tuple(df.dihedral.values[0])
    try:
        parameter = labels["ProperTorsions"][dihedral]
    except KeyError:
        try:
            parameter = labels["ProperTorsions"][dihedral[::-1]]
        except KeyError:
            raise ValueError(
                f"Could not find parameter for dihedral rotation {dihedral}."
            )

    all_dfs = []
    for _, row in tqdm.tqdm(
        df.iterrows(),
        total=len(df),
        desc="Calculating individual energy breakdown per conformer",
    ):
        conformer = np.array(row[coordinate_column]).reshape((-1, 3))
        mol._conformers = [conformer * unit.angstrom]

        df_ = calculate_energy_breakdown(mol, labels)
        df_["qcarchive_id"] = row.qcarchive_id
        df_["grid_id"] = row.grid_id
        df_["mapped_smiles"] = row.mapped_smiles
        df_["forcefield"] = forcefield
        all_dfs.append(df_)
    
    all_dfs = pd.concat(all_dfs)

    output_directory = pathlib.Path(output_directory)
    ff_name = pathlib.Path(forcefield).stem
    energy_output_directory = (
        output_directory
        / ff_name
        / parameter.id
        / str(torsiondrive_id)
        / f"individual-energy-breakdowns-{geometry}"
    )
    output_file = energy_output_directory / "individual-energy-breakdowns.csv"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    all_dfs.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    mol._conformers = None
    print("Plotting individual energy breakdowns...")
    image_directory = energy_output_directory / "images"
    for parameter_type, subdf in all_dfs.groupby("parameter_type"):
        for atom_indices, subsubdf in tqdm.tqdm(
            subdf.groupby("atom_indices"),
            total=len(subdf.atom_indices.unique()),
            desc=f"Plotting {parameter_type} energy breakdowns"                      
        ):
            plot_minimization_energies(
                subsubdf,
                mol,
                atom_indices=atom_indices,
                parameter_type=parameter_type,
                output_directory=image_directory,
            )



if __name__ == "__main__":
    main()

