"""Hi-C tool: draw the contact map around a gene or TAD, with TAD borders marked."""
import cooler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from config import DATA
from tools.common import q, resolve_target, save_plot

HIC_FILE = DATA / "hic.mcool"  # copy your lab's Hi-C file here, under this name


def hic_view(target: str, resolution: int = 25000, flank_kb: int = 500) -> dict:
    """Draw the Hi-C contact map around a gene (or a TAD id), with the TAD borders drawn as squares.

    target: gene symbol or TAD id.  resolution: bin size in bp; it must exist in the .mcool file.
    flank_kb: how much to show on each side of the TAD. Dark squares on the diagonal = TADs.
    Returns the region shown and the path of the heatmap image.
    """
    tad, err = resolve_target(target)
    if err:
        return err
    t = q("SELECT DISTINCT tad_id, chrom, tad_start, tad_end FROM tad_genes WHERE tad_id = ?", [tad["tad_id"]]).iloc[0]
    try:
        clr = cooler.Cooler(f"{HIC_FILE}::resolutions/{resolution}")
    except Exception:
        return {"error": f"No Hi-C matrix at {resolution} bp.",
                "available": cooler.fileops.list_coolers(str(HIC_FILE))}

    chrom = t["chrom"] if t["chrom"] in clr.chromnames else t["chrom"].replace("chr", "")
    if chrom not in clr.chromnames:
        return {"error": f"{t['chrom']} is not in the Hi-C file (it has {clr.chromnames[:5]}...)."}
    start = max(0, int(t["tad_start"]) - flank_kb * 1000)
    end = min(int(clr.chromsizes[chrom]), int(t["tad_end"]) + flank_kb * 1000)
    balance = "weight" in clr.bins().columns  # use normalised (balanced) counts when available
    mat = clr.matrix(balance=balance).fetch(f"{chrom}:{start}-{end}")
    if not np.nanmax(mat, initial=0) > 0:
        return {"error": f"The Hi-C file has no contacts in {chrom}:{start}-{end}."}

    fig, ax = plt.subplots(figsize=(5.5, 5))
    extent = (start / 1e6, end / 1e6, end / 1e6, start / 1e6)
    im = ax.imshow(mat, cmap="Reds", norm=LogNorm(), extent=extent, interpolation="none")
    borders = q("""SELECT DISTINCT tad_id, tad_start, tad_end FROM tad_genes
                   WHERE chrom = ? AND tad_end > ? AND tad_start < ?""", [t["chrom"], start, end])
    for b in borders.itertuples():
        s, e = b.tad_start / 1e6, b.tad_end / 1e6
        ax.add_patch(Rectangle((s, s), e - s, e - s, fill=False, lw=1.5 if b.tad_id == t["tad_id"] else 0.8,
                               edgecolor="black" if b.tad_id == t["tad_id"] else "gray"))
    ax.set_xlabel(f"{t['chrom']} position (Mb)")
    ax.set_title(f"Hi-C around {target}, {resolution // 1000} kb bins\n{t['tad_id']} in black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="contacts" + (" (balanced)" if balance else ""))
    return {"target": target, "tad_id": t["tad_id"], "region": f"{t['chrom']}:{start}-{end}",
            "resolution_bp": resolution, "n_tads_shown": len(borders),
            "plot": save_plot(fig, f"hic_{t['tad_id']}_{resolution}")}


if __name__ == "__main__":
    print(hic_view("EGFR"))
