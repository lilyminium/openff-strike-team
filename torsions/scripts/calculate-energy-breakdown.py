import itertools

from openff.toolkit import Molecule, ForceField
from openff.units import unit

import numpy as np
import pandas as pd
import MDAnalysis as mda


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
