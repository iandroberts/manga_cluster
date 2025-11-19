from pathlib import Path
import sys

from astropy.table import Table
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import umap

def normalize_spectrum(wav, flux):
    spec_reg = (wav > 5500) & (wav < 5800)
    scale = np.median(flux[spec_reg])
    if scale == 0:
        return
    return flux / scale

def get_rmag(plateifu):
    home = Path.home()
    drppath = f"{str(home)}/sas/dr17/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"
    drpall = Table.read(drppath, hdu=1)
    row = drpall[drpall["plateifu"] == f"{plateifu}"][0]
    rmag = row["nsa_elpetro_absmag"][4]
    return rmag

def create_feature_vectors():
    dir = Path("./preprocessed_spectra")
    features = []
    for fpath in tqdm(dir.glob("*.fits")):
        continue_outer = False
        t = Table.read(f"{fpath}")
        N = len(t)
        feat = np.zeros(6*N+1)
        for i in range(len(t.colnames) - 1):
            fnorm = normalize_spectrum(t["wav"], t[f"flux{i+1}"])
            if fnorm is None:
                continue_outer = True
                break
            feat[N*i:N*(i+1)] = fnorm
        if continue_outer:
            continue
        plateifu = Path(fpath).stem
        absmag = get_rmag(plateifu)
        feat[-1] = absmag
        features.append(feat)
    features = np.array(features)
    features[:, -1] = features[:, -1] - np.median(features[:, -1])
    return features

def run_PCA(features, ncomp=30, plot_scree=False, plot_comp=False):
    pca = PCA(n_components=ncomp, random_state=42)
    lowd = pca.fit_transform(features)
    totvar = np.sum(pca.explained_variance_ratio_)
    print(totvar)

    if plot_scree:
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(cumvar, "ko-")
        plt.show()

    if plot_comp:
        for c in pca.components_:
            plt.plot(c, "k-")
            plt.show()
            plt.close()

    return lowd

def run_umap(features, **kwargs):
    um = umap.UMAP(**kwargs)
    embedding = um.fit_transform(features)
    return embedding

def plot_umap(features, labels):
    um = umap.UMAP(n_components=2, random_state=42)
    embedding = um.fit_transform(features)
    clustered = (labels >= 0)
    plt.scatter(embedding[~clustered, 0],
            embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=50,
            alpha=0.5)
    plt.scatter(embedding[clustered, 0],
            embedding[clustered, 1],
            c=labels[clustered],
            s=50,
            alpha=0.75,
            cmap='Spectral')
    plt.show()

def run_hdbscan(features, **kwargs):
    hd = hdbscan.HDBSCAN(**kwargs)
    labels = hd.fit_predict(features)
    return labels

if __name__ == "__main__":
    features = create_feature_vectors()
    lowd_features = run_PCA(features)

    umap_features = run_umap(
        lowd_features,
        n_neighbors=15,
        min_dist=0.0,
        n_components=10,
        random_state=42,
    )

    labels = run_hdbscan(
        umap_features,
        min_samples=2,
        min_cluster_size=10,
        approx_min_span_tree=False
    )

    print(labels)
    plot_umap(lowd_features, labels)
