import tqdm
import click
import json
from collections import defaultdict

from openff.toolkit import Molecule, ForceField

@click.command()
@click.option(
    "--input",
    "input_dataset_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="The path to the input Parquet path of the torsiondrive dataset. ",
    default="qm/output"
)
@click.option(
    "--force-field",
    "forcefield",
    type=str,
    help="The name or path of the force field to use to optimize in the database.",
)
@click.option(
    "--output",
    "output_file",
    type=click.Path(exists=False, dir_okay=False, file_okay=True),
    default="parameter_id_to_torsion_ids.json"
)
def main(
    input_dataset_path: str,
    forcefield: str,
    output_file: str,
):
    import pyarrow.dataset as ds

    dataset = ds.dataset(input_dataset_path, format="parquet")
    ff = ForceField(forcefield, allow_cosmetic_attributes=True)
    columns = ['mapped_smiles', 'torsiondrive_id', 'dihedral']
    df = dataset.to_table(columns=columns).to_pandas()
    df["dihedral"] = [tuple(x) for x in df["dihedral"]]
    unique = df.groupby(by=columns).first().reset_index()

    parameter_id_to_torsion_ids = defaultdict(list)
    errors = []

    for _, row in tqdm.tqdm(unique.iterrows(), total=len(unique)):
        torsion_id = row["torsiondrive_id"]
        mol = Molecule.from_mapped_smiles(
            row["mapped_smiles"],
            allow_undefined_stereo=True
        )
        labels = ff.label_molecules(mol.to_topology())[0]["ProperTorsions"]
        dihedral = tuple(row["dihedral"])
        if dihedral in labels:
            parameter_id = labels[dihedral].id
        elif dihedral[::-1] in labels:
            parameter_id = labels[dihedral[::-1]].id
        else:
            errors.append(dict(row))

        parameter_id_to_torsion_ids[parameter_id].append(torsion_id)

    with open(output_file, "w") as f:
        json.dump(parameter_id_to_torsion_ids, f)
    print(f"Saved {output_file}")

    print(
        f"Found {len(errors)} errors where dihedral "
        f"cannot be labelled by {forcefield}."
    )
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
