import click

import pathlib
import pickle

def draw(df, output_file: str, n_col: int = 4, n_page: int = 24):
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import Draw
    from openff.toolkit.topology import Molecule
    
    rdmols = []
    legends = []
    for _, row in df.iterrows():
        mol = Molecule.from_smiles(
            row["smiles"],
            allow_undefined_stereo=True
        )
        rdmol = mol.to_rdkit()
        rdmols.append(rdmol)
        legends.append(f"{row['cluster']}")

    images = []
    for i in range(0, len(rdmols), n_page):
        j = i + n_page
        rdmols_chunk = rdmols[i:j]
        legends_chunk = legends[i:j]
        
        img = Draw.MolsToGridImage(
            rdmols_chunk,
            molsPerRow=n_col,
            legends=legends_chunk,
            subImgSize=(300, 300),
            maxMols=n_page,
            returnPNG=False,
        )
        
        images.append(img)
    images[0].save(output_file, append_images=images[1:], save_all=True, dpi=(300, 300))
    print(f"Saved {output_file}")


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default="../labels/fingerprints/morgan/fingerprints",
)
@click.option(
    "--distance-matrix",
    "distance_matrix_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    default="../labels/fingerprints/morgan/dissimilarity_matrix.npy",
)
@click.option(
    "--fingerprint",
    "fingerprint_type",
    type=str,
    default="morgan",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, dir_okay=True, file_okay=False),
    default="../labels/fingerprints/morgan/clusters/cutoff-0.30"
)
@click.option(
    "--cutoff",
    type=float,
    default=0.3,
)
def main(
    input_path: str,
    distance_matrix_path: str,
    fingerprint_type: str,
    output_path: str,
    cutoff: float = 0.05,
    
):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    
    from rdkit.ML.Cluster import Butina

    dataset = ds.dataset(input_path)
    expression = pc.field("name") == fingerprint_type
    subset = dataset.filter(expression)
    n_fingerprints = subset.count_rows()

    distance_matrix = np.load(distance_matrix_path)

    clusters = Butina.ClusterData(
        distance_matrix,
        n_fingerprints,
        cutoff,
        isDistData=True,
    )
    del distance_matrix
    clusters = sorted(clusters, key=len, reverse=True)
    print(f"Number of clusters: {len(clusters)}")

    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cluster_indices_file = output_path / "cluster_indices.pkl"
    with open(cluster_indices_file, "wb") as f:
        pickle.dump(clusters, f)

    data_rows = [
        dict(row) for row in
        dataset.to_table().to_pylist()
    ]

    row_counts = []

    for cluster_index, row_indices in enumerate(clusters):
        row_counts.append({"cluster": cluster_index, "count": len(row_indices)})
        for row_index in row_indices:
            data_rows[row_index]["cluster"] = cluster_index
            data_rows[row_index]["centroid"] = False
        data_rows[row_indices[0]]["centroid"] = True
    
    new_table = pa.Table.from_pylist(data_rows)
    output_dataset = output_path / "dataset_with_clusters"
    ds.write_dataset(new_table, output_dataset, format="parquet")

    centroid_expression = pc.field("centroid")
    centroids = new_table.filter(centroid_expression)
    df = centroids.to_pandas()
    centroid_file = output_path / "centroids.csv"
    df.to_csv(centroid_file, index=False)
    print(f"Centroids saved to {centroid_file}")

    count_df = pd.DataFrame(row_counts)
    count_file = output_path / "cluster_counts.csv"
    count_df.to_csv(count_file, index=False)

    df = df.sort_values("cluster")
    draw(df, output_path / "clusters.png")


if __name__ == "__main__":
    main()
