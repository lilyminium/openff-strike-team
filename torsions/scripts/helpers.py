from rdkit import Chem
import pathlib
import itertools
import tempfile
import tqdm
import os
import json

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

def write_pdbs(
    qcarchive_id: int,
    forcefield: str = "openff-2.2.0-rc1.offxml",
    input_dataset: str = "/Volumes/OpenFF/hpc3/benchmarking/07_general-qm/minimized-trajectories",
    output_directory: str = "minimizations"
):
    dataset = ds.dataset(input_dataset, format="parquet")
    
    ff_expression = pc.field("forcefield") == forcefield
    subset = dataset.filter(ff_expression)

    expression = pc.field("qcarchive_id") == qcarchive_id
    subset = subset.filter(expression)
    table = subset.to_table(columns=["forcefield", "mapped_smiles", "Iteration", "mm_coordinates"])
    rows = table.to_pylist()

    coordinates = []
    for i, row in enumerate(rows):
        assert i == row["Iteration"]
        coordinates.append(
            np.array(row["mm_coordinates"]).reshape((-1, 3)) * 10
        )

    mol = Molecule.from_mapped_smiles(row["mapped_smiles"])
    u = mda.Universe(mol.to_rdkit())
    u.load_new(np.array(coordinates))

    output_directory = pathlib.Path(output_directory) / pathlib.Path(forcefield).stem
    output_directory.mkdir(exist_ok=True, parents=True)
    file = output_directory / f"{qcarchive_id}.pdb"
    with mda.Writer(file, len(u.atoms)) as writer:
        for _ in u.trajectory:
            writer.write(u.atoms)
    print(f"Wrote {len(u.trajectory)} frames to {file}")

def draw_grid_df(
    df,
    use_svg: bool = True,
    output_file: str = None,
    n_col: int = 4,
    n_page: int = 24,
    subImgSize=(300, 300),
):
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import Draw
    from openff.toolkit import Molecule
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF, renderPM
    import tempfile
    import os

    
    rdmols = []
    legends = []
    tags = []
    for _, row in df.iterrows():
        indices = row["atom_indices"]
        mol = Molecule.from_mapped_smiles(
            row["mapped_smiles"],
            allow_undefined_stereo=True
        )
        rdmol = mol.to_rdkit()
        for index in indices:
            atom = rdmol.GetAtomWithIdx(int(index))
            atom.SetProp("atomNote", str(index))
        rdmols.append(rdmol)
        indices_text = "-".join(list(map(str, indices)))
        legends.append(f"{row['torsiondrive_id']}: {indices_text}")
        tags.append(tuple(map(int, indices)))

    images = []
    for i in range(0, len(rdmols), n_page):
        j = i + n_page
        rdmols_chunk = rdmols[i:j]
        legends_chunk = legends[i:j]
        tags_chunk = tags[i:j]
        
        img = Draw.MolsToGridImage(
            rdmols_chunk,
            molsPerRow=n_col,
            legends=legends_chunk,
            subImgSize=subImgSize,
            maxMols=n_page,
            highlightAtomLists=tags_chunk,
            returnPNG=False,
            useSVG=use_svg,
        )
        
        images.append(img)
    if output_file:
        output_file = pathlib.Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        if not use_svg:
            images[0].save(output_file, append_images=images[1:], save_all=True, dpi=(300, 300))
            print(f"Saved {output_file}")
        else:
            base_file, suffix = str(output_file).rsplit(".", maxsplit=1)
            for i, img in enumerate(images):
                file = f"{base_file}_{i}.{suffix}"

                with tempfile.TemporaryDirectory() as tempdir:
                    cwd = os.getcwd()
                    os.chdir(tempdir)

                    with open("temp.svg", "w") as f:
                        f.write(img.data)
                    drawing = svg2rlg("temp.svg")
                os.chdir(cwd)
                renderPM.drawToFile(drawing, file, fmt="PNG")
                print(f"Saved {file}")

    return images


def display(smiles: str):
    """Display smiles or molecule"""
    if isinstance(smiles, str):
        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    else:
        mol = smiles
    rdmol = mol.to_rdkit()
    for atom in rdmol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return rdmol


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


def calculate_energy_breakdown(mol, forcefield):
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    labels = ff.label_molecules(mol.to_topology())[0]
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
                "energy": energy.m_as(unit.kilojoules_per_mole),
                "forcefield": forcefield,
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
                "energy": energy.m_as(unit.kilojoules_per_mole),
                "forcefield": forcefield,
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
            "energy": energy.m_as(unit.kilojoules_per_mole),
            "forcefield": forcefield,
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
            "energy": energy.m_as(unit.kilojoules_per_mole),
            "forcefield": forcefield,
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
            "energy": energy.m_as(unit.kilojoules_per_mole),
            "forcefield": forcefield,
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
            "energy": energy.m_as(unit.kilojoules_per_mole),
            "forcefield": forcefield,
        }
        all_entries.append(entry)



    df = pd.DataFrame(all_entries)
    return df
            

def get_minimization_energies(
    qcarchive_id: int,
    forcefield: str = "openff-2.2.0-rc1.offxml",
    output_directory = "single-molecules",
):
    dataset = ds.dataset("/Volumes/OpenFF/hpc3/benchmarking/07_general-qm/minimized-trajectories", format="parquet")
    output_directory = pathlib.Path(output_directory) / pathlib.Path(forcefield).stem / str(qcarchive_id)
    output_directory.mkdir(exist_ok=True, parents=True)

    ff_expression = pc.field("forcefield") == forcefield
    subset = dataset.filter(ff_expression)

    expression = pc.field("qcarchive_id") == qcarchive_id
    subset = subset.filter(expression)
    table = subset.to_table(columns=["mapped_smiles", "Iteration", "mm_coordinates"])
    df = table.to_pandas().sort_values("Iteration", ascending=True)

    dfs = []
    mol = Molecule.from_mapped_smiles(
        df["mapped_smiles"].values[0],
        allow_undefined_stereo=True
    )
    mol.assign_partial_charges("am1bccelf10")
    for _, row in tqdm.tqdm(df.iterrows()):
        mol._conformers = [np.array(row["mm_coordinates"]).reshape((-1, 3)) * unit.nanometers]
        df_ = calculate_energy_breakdown(mol, forcefield)
        df_["Iteration"] = row["Iteration"]
        dfs.append(df_)

    output_df = pd.concat(dfs)
    output_df["qcarchive_id"] = qcarchive_id
    output_df["mapped_smiles"] = df["mapped_smiles"].values[0]


    file = output_directory / "parameter-minimizations.csv"
    output_df.to_csv(file)
    print(f"Saved to {file}")
    return output_df


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
    ax2.set_ylabel("Energy (kJ/mol)", color=color2)
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


def plot_all_minimization_energies(
    df,
    output_directory = "single-molecules",
):
    for parameter_type in df.parameter_type.unique():
        subdf = df[df.parameter_type == parameter_type]
        for atom_indices in subdf.atom_indices.unique():
            plot_minimization_energies(
                subdf,
                atom_indices=atom_indices,
                parameter_type=parameter_type,
                output_directory=output_directory,
            )
    first_iter = df[df.Iteration == 0]
    forcefield = subdf.forcefield.values[0]
    qcarchive_id = subdf.qcarchive_id.values[0]
    output_directory = (
        pathlib.Path(output_directory)
        / pathlib.Path(forcefield).stem
        / f"{qcarchive_id}/images/parameter-indices"
    )
    output_directory.mkdir(exist_ok=True, parents=True)
    file = str(output_directory / "all-parameter-indices.png")

    draw_grid_df(
        first_iter,
        output_file=file,
        use_svg=True,
        n_col=4,
        n_page=len(first_iter),
        subImgSize=(300, 300),
    )


def plot_grouped_minimization_energies_singlepoint(
    torsiondrive_id: int,
    forcefield: str = "openff-2.2.0-rc1.offxml",
    output_directory = "single-molecules",
):
    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")
    subset = dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)

    minimized = ds.dataset("../../benchmarking/07_general-qm/singlepoint-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)
    minimized = minimized.filter(pc.field("torsiondrive_id") == torsiondrive_id)
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
    # geometry_df = minimized.filter(
    #     pc.field("qcarchive_id").isin(qca_ids_df.qcarchive_id.values)
    # ).to_table(
    #     columns=[
    #         "qcarchive_id",
    #         "Bond",
    #         "Angle",
    #         "Torsion",
    #         "vdW",
    #         "Electrostatics",
    #         "vdW 1-4",
    #         "Electrostatics 1-4",
    #         "mm_energy"
    #     ]
    # ).to_pandas()

    df = qca_ids_df.merge(geometry_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="inner")
    df["Total"] = df["mm_energy"]
    # df["Bond"] = df["energy_Bond"]
    # df["Angle"] = df["energy_Angle"]
    # df["Torsion"] = df["energy_Torsion"]
    # df["Nonbonded"] = df["energy_Nonbonded"]
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
    file = output_directory / "grouped-minimizations.png"
    g.savefig(file, dpi=300)
    print(f"Saved to {file}")


def plot_grouped_minimization_energies(
    torsiondrive_id: int,
    forcefield: str = "openff-2.2.0-rc1.offxml",
    output_directory = "single-molecules",
):
    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")

    minimized = ds.dataset("../../benchmarking/07_general-qm/minimized-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)

    subset = dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    qca_ids_df = subset.to_table().to_pandas()
    geometry_df = minimized.filter(
        pc.field("qcarchive_id").isin(qca_ids_df.qcarchive_id.values)
    ).to_table(
        columns=[
            "qcarchive_id",
            "energy_Bond",
            "energy_Angle",
            "energy_Torsion",
            "energy_Nonbonded",
            "mm_energy"
        ]
    ).to_pandas()

    df = qca_ids_df.merge(geometry_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="inner")
    df["Total"] = df["mm_energy"]
    df["Bond"] = df["energy_Bond"]
    df["Angle"] = df["energy_Angle"]
    df["Torsion"] = df["energy_Torsion"]
    df["Nonbonded"] = df["energy_Nonbonded"]
    df = df.sort_values("grid_id")
    

    melted = df.melt(
        id_vars=["grid_id", "qcarchive_id"],
        value_vars=[
            "Angle", "Bond",
            "Torsion", "Nonbonded", "Total"
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
    file = output_directory / "grouped-minimizations.png"
    g.savefig(file, dpi=300)
    print(f"Saved to {file}")


def get_and_plot_minimization_energies(
    qcarchive_id: int,
    forcefield: str = "openff-2.2.0-rc1.offxml",
    output_directory = "single-molecules",
):
    qcarchive_id_directory = (
        pathlib.Path(output_directory)
        / pathlib.Path(forcefield).stem
        / str(qcarchive_id)
    )
    qcarchive_id_directory.mkdir(exist_ok=True, parents=True)
    plot_grouped_minimization_energies(qcarchive_id, forcefield, output_directory)

    df = get_minimization_energies(qcarchive_id, forcefield, output_directory)
    plot_all_minimization_energies(df, output_directory)
    write_pdbs(qcarchive_id=qcarchive_id, forcefield=forcefield, output_directory=qcarchive_id_directory)
    return df


def plot_with_torsion(parameter_id: str, forcefield: str, parameter_id_to_torsion_ids: dict[str, list[int]]):
    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    handler = ff.get_parameter_handler("ProperTorsions")
    
    output_directory = pathlib.Path("images") / parameter_id
    output_directory.mkdir(exist_ok=True, parents=True)
    
    parameter = handler.get_parameter({"id": parameter_id})[0]
    smirks = parameter.smirks
    torsiondrive_ids = parameter_id_to_torsion_ids[parameter_id]

    expression = pc.field("torsiondrive_id").isin(torsiondrive_ids)
    subset = dataset.filter(expression)
    cols = ["torsiondrive_id", "grid_id", "energy", "mapped_smiles", "dihedral"]
    df = subset.to_table(columns=cols).to_pandas()
    df["atom_indices"] = [tuple(x) for x in df["dihedral"]]

    subdfs = []
    for tid, subdf in df.groupby("torsiondrive_id"):
        relative_energy = subdf["energy"] - subdf["energy"].min()
        subdf["relative_energy"] = relative_energy
        subdfs.append(subdf)

    df = pd.concat(subdfs)

    df = df.sort_values(by=["torsiondrive_id", "grid_id"])

    draw_df = df.groupby("torsiondrive_id").first().reset_index()
    images = draw_grid_df(draw_df, output_file=output_directory / "molecules.png")
    
    rows = []
    for angle in range(-180, 181, 15):
        energy = calc_torsion_energy(angle, parameter).m_as(unit.kilocalories_per_mole)
        rows.append({
            "torsiondrive_id": "Sage 2.2 TM",
            "grid_id": angle,
            "energy": energy,
            "relative_energy": energy,
            "mapped_smiles": "",
            "dihedral": [],
            "atom_indices": tuple()
        })
    df2 = pd.DataFrame(rows)
    df = pd.concat([df2, df])

    g = sns.FacetGrid(
        data=df,
        hue="torsiondrive_id",
        height=4,
        aspect=1.4,
    )
    g.map(sns.lineplot, "grid_id", "relative_energy")
    ax = list(g.axes.flatten())[0]
    ax.set_title(f"{parameter_id}\n{smirks}")
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Relative energy\n[kcal/mol]")
    plt.tight_layout()
    
    g.add_legend()
    g.savefig(output_directory / "torsions.png", dpi=300)
    
    
    
    return ax, images



def plot_with_mm(parameter_id: str, forcefield: str, parameter_id_to_torsion_ids: dict[str, list[int]], output_directory: str = "images"):
    
    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")
    # dataset = dataset.filter(pc.field("forcefield") == forcefield)
    minimized = ds.dataset("../../benchmarking/07_general-qm/minimized-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    handler = ff.get_parameter_handler("ProperTorsions")
    
    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)
    
    parameter = handler.get_parameter({"id": parameter_id})[0]
    smirks = parameter.smirks
    torsiondrive_ids = parameter_id_to_torsion_ids[parameter_id]
    if len(torsiondrive_ids) == 0:
        return

    expression = pc.field("torsiondrive_id").isin(torsiondrive_ids)
    subset = dataset.filter(expression)
    cols = ["torsiondrive_id", "grid_id", "energy", "mapped_smiles", "dihedral", "qcarchive_id"]
    df = subset.to_table(columns=cols).to_pandas()
    df["atom_indices"] = [tuple(x) for x in df["dihedral"]]

    draw_df = df.groupby("torsiondrive_id").first().reset_index()
    images = draw_grid_df(
        draw_df,
        output_file=output_directory / "molecules" / "molecules.png",
        subImgSize=(500, 500)
    )

    subdfs = []

    minimized_expression = pc.field("qcarchive_id").isin(df.qcarchive_id.values)
    minimized_subset = minimized.filter(minimized_expression)
    minimized_energies = minimized_subset.to_table(columns=["qcarchive_id", "mm_energy"])
    minimized_df = minimized_energies.to_pandas().sort_values("qcarchive_id")

    if len(minimized_df) == 0:
        print(f"{parameter_id} has no minimized geometries")
        df["mm_energy"] = np.nan
    else:
        df = df.sort_values("qcarchive_id")
        df = df.merge(minimized_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="left")
        # df["mm_energy"] = minimized_df["mm_energy"]

        # assert len(df) == len(minimized_df)
        

    # remove accidental duplicates
    df = df.groupby(by=["torsiondrive_id", "grid_id", "atom_indices", "qcarchive_id"]).first().reset_index()

    for tid, subdf in df.groupby("torsiondrive_id"):
        relative_qm_energy = subdf["energy"] - subdf["energy"].min()
        subdf["relative_qm_energy"] = relative_qm_energy
        relative_mm_energy = subdf["mm_energy"] - subdf["mm_energy"].min()
        subdf["relative_mm_energy"] = relative_mm_energy
        subdfs.append(subdf)

    df = pd.concat(subdfs)
    df = df.melt(
        id_vars=[x for x in df.columns if x not in ["relative_qm_energy", "relative_mm_energy"]],
        value_vars=["relative_qm_energy", "relative_mm_energy"],
        var_name="Type",
        value_name="relative_energy",
    )
    
    df = df.sort_values(by=["torsiondrive_id", "grid_id"])
    df["TorsionDrive ID"] = [str(x) for x in df.torsiondrive_id.values]

    plt.clf()
    g = sns.FacetGrid(data=df, aspect=1.4, height=4, hue="TorsionDrive ID")
    g.map_dataframe(sns.lineplot, "grid_id", "relative_energy", style="Type")
    # g.map_dataframe(map_lineplot)

    ax = list(g.axes.flatten())[0]

    title = f"{parameter_id}\n" + smirks.encode("unicode_escape").decode("utf-8").replace("$", "\$")
    print(title)
    ax.set_title(title)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Relative energy\n[kcal/mol]")
    plt.tight_layout()
    
    g.add_legend()
    filename = output_directory / "mm-torsion-energies.png"
    g.savefig(filename, dpi=300)
    print(f"Saved to {filename}")
    return ax
    
    
    

def draw_single(df, torsiondrive_id: int, width=300, height=300):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from openff.toolkit import Molecule
    from rdkit.Chem.Draw import rdMolDraw2D
    from matplotlib import pyplot as plt
    from cairosvg import svg2png
    
    subdf = df[df.torsiondrive_id == torsiondrive_id]
    mol = Molecule.from_mapped_smiles(
        subdf["mapped_smiles"].values[0],
        allow_undefined_stereo=True
    )
    rdmol = mol.to_rdkit()
    dihedral = list(map(int, subdf["dihedral"].values[0]))
    for index in dihedral:
        atom = rdmol.GetAtomWithIdx(int(index))
        atom.SetProp("atomNote", str(index))
    indices = "-".join(list(map(str, dihedral)))
    
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    # options.baseFontSize = 1
    drawer.DrawMolecule(rdmol, highlightAtoms=dihedral)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        svg2png(bytestring=svg, write_to='tmp.png', scale=10)
        png = plt.imread("tmp.png")
        os.chdir(cwd)
    
    return png

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

def plot_mm_vs_qm_profile(
    torsiondrive_id: int, output_directory=None, forcefield: str = "tm-2.2.offxml",
    with_rmsds: bool = True,
    minimized_geometry: bool = True
):
    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")
    # dataset = dataset.filter(pc.field("forcefield") == forcefield)
    if minimized_geometry:
        minimized = ds.dataset("../../benchmarking/07_general-qm/minimized-torsiondrive-datasets")
    else:
        minimized = ds.dataset("../../benchmarking/07_general-qm/singlepoint-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)
    
    subset = dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    df = subset.to_table().to_pandas()
    columns = ["qcarchive_id", "qm_energy", "mm_energy"]
    if with_rmsds:
        columns.append("RMSD_AA")
    geometry_df = minimized.filter(
        pc.field("qcarchive_id").isin(df.qcarchive_id.values)
    ).to_table(columns=columns).to_pandas()

    joined = df.merge(geometry_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="inner")
    if not len(joined):
        print(f"{torsiondrive_id} has no minimized geometries: {len(df)} QM")
        return
    min_index = np.argmin(joined.qm_energy.values)
    joined["QM"] = joined.qm_energy - joined.qm_energy.values[min_index]
    joined["MM"] = joined.mm_energy - joined.mm_energy.values[min_index]
    joined = joined.sort_values("grid_id")

    fig, (ax1, imgax) = plt.subplots(figsize=(8, 4), ncols=2)
    color1 = "#015480"
    color1b = "#2F9ED2"
    ax1.set_xlabel("Angle")
    ax1.set_ylabel("Energy [kcal/mol]", color=color1)
    ax1.plot(joined.grid_id, joined.QM, color=color1, label="QM")
    ax1.plot(joined.grid_id, joined.MM, color=color1b, label="MM")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.legend()

    if with_rmsds:
        ax2 = ax1.twinx()
        color2 = "#F03A21"
        ax2.set_ylabel("RMSD [Å]", color=color2)
        ax2.plot(joined.grid_id, joined.RMSD_AA, color=color2, label="RMSD")
        ax2.axhline(0.4, color=color2, ls="--", lw=1)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax2.set_title(f"{torsiondrive_id}")
    else:
        ax1.set_title(f"{torsiondrive_id}")
    
    png = draw_single(df, torsiondrive_id)
    imgax.imshow(png, rasterized=True)
    imgax.set_xticks([])
    imgax.set_yticks([])
    imgax.spines["left"].set_visible(False)
    imgax.spines["right"].set_visible(False)
    imgax.spines["top"].set_visible(False)
    imgax.spines["bottom"].set_visible(False)
    
    fig.tight_layout()

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    imgfile = output_directory / f"{torsiondrive_id}.png"
    plt.savefig(imgfile, dpi=300)
    print(f"Saved to {imgfile}")


def analyze_torsions(forcefield):
    output_directory = pathlib.Path("images")
    with open("parameter_id_to_torsion_ids.json", "r") as f:
        parameter_id_to_torsion_ids = json.load(f)

    start = False
    parameter = "t79"
    for k in tqdm.tqdm(sorted(parameter_id_to_torsion_ids, key=lambda x: len(parameter_id_to_torsion_ids[x]))):
        if k == parameter:
            start = True
        if not start:
            continue
        image_directory = output_directory / k
        # plot_with_mm(k, forcefield, parameter_id_to_torsion_ids, image_directory)
        for torsion_id in tqdm.tqdm(parameter_id_to_torsion_ids[k], desc=k):
            # plot_mm_vs_qm_profile(torsion_id, image_directory / "torsions", forcefield)
            plot_mm_vs_qm_profile(
                torsion_id, image_directory / "singlepoints", forcefield,
                with_rmsds=False, minimized_geometry=False
            )


def plot_torsiondrive_minimization(torsiondrive_id: int, forcefield: str, plot_all: bool = True):
    output_directory = pathlib.Path("images")

    dataset = ds.dataset("../../datasets/qm/output/torsiondrive")
    minimized = ds.dataset("../../benchmarking/07_general-qm/minimized-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)
    
    subset = dataset.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    df = subset.to_table().to_pandas()
    geometry_df = minimized.filter(
        pc.field("qcarchive_id").isin(df.qcarchive_id.values)
    ).to_table(
        columns=["qcarchive_id", "qm_coordinates", "mm_coordinates"]
    ).to_pandas()

    joined = df.merge(geometry_df, left_on=["qcarchive_id"], right_on=["qcarchive_id"], how="inner")
    if not len(joined):
        print(f"{torsiondrive_id} has no minimized geometries: {len(df)} QM")
        return
    
    joined = joined.sort_values("grid_id")

    mol = Molecule.from_mapped_smiles(
        joined["mapped_smiles"].values[0],
        allow_undefined_stereo=True
    )
    mol.assign_partial_charges("am1bccelf10")

    # get torsion
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    labels = ff.label_molecules(mol.to_topology())[0]["ProperTorsions"]
    indices = tuple(joined["dihedral"].values[0])
    if indices not in labels:
        indices = tuple(reversed(indices))
    parameter = labels[indices]

    dfs = []
    for _, row in tqdm.tqdm(joined.iterrows(), desc="Calculating energies"):
        mol._conformers = [np.array(row["mm_coordinates"]).reshape((-1, 3)) * unit.angstrom]
        df_ = calculate_energy_breakdown(mol, forcefield)
        df_["grid_id"] = row["grid_id"]
        dfs.append(df_)
    output_df = pd.concat(dfs)
    output_df["torsiondrive_id"] = torsiondrive_id

    torsiondrive_directory = (
        output_directory
        / parameter.id
        / pathlib.Path(forcefield).stem
        / "torsion-energies"
        / str(torsiondrive_id)
    )
    torsiondrive_directory.mkdir(exist_ok=True, parents=True)
    filename = torsiondrive_directory / "parameter-minimizations.csv"
    output_df.to_csv(filename)
    print(f"Saved to {filename}")

    plot_grouped_minimization_energies(torsiondrive_id, forcefield, torsiondrive_directory)

    if plot_all:
        mol._conformers = []

        parameter_directory = torsiondrive_directory / "parameters"
        parameter_directory.mkdir(exist_ok=True, parents=True)
        
        for parameter_type in output_df.parameter_type.unique():
            subdf = output_df[output_df.parameter_type == parameter_type]
            for atom_indices in tqdm.tqdm(
                subdf.atom_indices.unique(),
                desc=parameter_type
            ):
                plot_minimization_energies(
                    subdf,
                    mol,
                    atom_indices=atom_indices,
                    parameter_type=parameter_type,
                    output_directory=parameter_directory,
                )


def plot_torsiondrive_singlepoint(torsiondrive_id: int, forcefield: str, plot_all: bool = True):
    output_directory = pathlib.Path("images")

    minimized = ds.dataset("../../benchmarking/07_general-qm/singlepoint-torsiondrive-datasets")
    minimized = minimized.filter(pc.field("forcefield") == forcefield)
    minimized = minimized.filter(pc.field("torsiondrive_id") == torsiondrive_id)
    
    joined = minimized.to_table().to_pandas()
        
    joined = joined.sort_values("grid_id")

    mol = Molecule.from_mapped_smiles(
        joined["mapped_smiles"].values[0],
        allow_undefined_stereo=True
    )
    mol.assign_partial_charges("am1bccelf10")

    # get torsion
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    labels = ff.label_molecules(mol.to_topology())[0]["ProperTorsions"]
    indices = tuple(joined["dihedral"].values[0])
    if indices not in labels:
        indices = tuple(reversed(indices))
    parameter = labels[indices]

    dfs = []
    for _, row in tqdm.tqdm(joined.iterrows(), desc="Calculating energies"):
        mol._conformers = [np.array(row["conformer"]).reshape((-1, 3)) * unit.angstrom]
        df_ = calculate_energy_breakdown(mol, forcefield)
        df_["grid_id"] = row["grid_id"]
        dfs.append(df_)
    output_df = pd.concat(dfs)
    output_df["torsiondrive_id"] = torsiondrive_id

    torsiondrive_directory = (
        output_directory
        / parameter.id
        / pathlib.Path(forcefield).stem
        / "singlepoint-energies"
        / str(torsiondrive_id)
    )
    torsiondrive_directory.mkdir(exist_ok=True, parents=True)
    filename = torsiondrive_directory / "parameter-minimizations.csv"
    output_df.to_csv(filename)
    print(f"Saved to {filename}")

    plot_grouped_minimization_energies_singlepoint(torsiondrive_id, forcefield, torsiondrive_directory)

    if plot_all:
        mol._conformers = []

        parameter_directory = torsiondrive_directory / "parameters"
        parameter_directory.mkdir(exist_ok=True, parents=True)
        
        for parameter_type in output_df.parameter_type.unique():
            subdf = output_df[output_df.parameter_type == parameter_type]
            for atom_indices in tqdm.tqdm(
                subdf.atom_indices.unique(),
                desc=parameter_type
            ):
                plot_minimization_energies(
                    subdf,
                    mol,
                    atom_indices=atom_indices,
                    parameter_type=parameter_type,
                    output_directory=parameter_directory,
                )

