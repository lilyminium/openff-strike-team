import click
import pathlib

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