import functools
import pathlib

import click
import tqdm
import multiprocessing

def convert_to_rdkit(fingerprint: str):
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect

    fp = ExplicitBitVect(2048)
    fp.FromBase64(fingerprint)
    return fp


# def compute_dissimilarity(i, fingerprints=None):
#     from rdkit import DataStructs

#     similarities = DataStructs.BulkTanimotoSimilarity(
#         fingerprints[i], fingerprints[:i]
#     )
#     return [1 - sim for sim in similarities]


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default="../labels/fingerprints/morgan/fingerprints",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../labels/fingerprints/morgan/dissimilarity_matrix.npy",
)
@click.option(
    "--fingerprint",
    "fingerprint_type",
    type=str,
    default="morgan",
)
def main(
    input_path: str,
    output_path: str,
    fingerprint_type: str,
):
    import pyarrow.dataset as ds
    import pyarrow.compute as pc
    import numpy as np
    from rdkit import DataStructs

    dataset = ds.dataset(input_path)
    expression = pc.field("name") == fingerprint_type
    subset = dataset.filter(expression)
    b64s = subset.to_table(
        columns=["fingerprint"]
    ).to_pydict()["fingerprint"]
    fingerprints = [
        convert_to_rdkit(b64)
        for b64 in tqdm.tqdm(b64s, desc="Converting fingerprints")
    ]
    n_fingerprints = len(fingerprints)
    print(f"Total number of fingerprints: {n_fingerprints}")

    n_dissimilarities = n_fingerprints * (n_fingerprints - 1) // 2
    dissimilarity_matrix = np.zeros(n_dissimilarities, dtype=float)

    counter = 0
    for i in tqdm.tqdm(
        range(1, n_fingerprints),
        total=n_fingerprints,
        desc="Computing distance matrix",
    ):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i], fingerprints[:i]
        )
        values = [1 - sim for sim in similarities]
        n_values = len(values)
        dissimilarity_matrix[counter : counter + n_values] = values
        counter += n_values

    np.save(output_path, dissimilarity_matrix)
    print(f"Distance matrix saved to {output_path}")


if __name__ == "__main__":
    main()
