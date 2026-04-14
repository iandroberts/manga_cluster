from pathlib import Path
import sys

from astropy.table import Table
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
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
        #feat = np.zeros(6*N+1)
        feat = np.zeros(6*N)
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
        #feat[-1] = absmag
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
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(embedding[~clustered, 0],
            embedding[~clustered, 1],
            #embedding[~clustered, 2],
            color=(0.5, 0.5, 0.5),
            s=9,
            edgecolors="none",
            alpha=0.5)
    ax.scatter(embedding[clustered, 0],
            embedding[clustered, 1],
            #embedding[clustered, 2],
            c=labels[clustered],
            s=25,
            edgecolors="none",
            alpha=0.75,
            cmap='Spectral')
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    fig.savefig("plots/umap.pdf", bbox_inches="tight", pad_inches=0.03)

def run_hdbscan(features, **kwargs):
    hd = hdbscan.HDBSCAN(**kwargs)
    labels = hd.fit_predict(features)
    return labels

def hdbscan_dbcv_score(data, min_cluster_size, min_samples, n_neighbors):
    try:
        mapper = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=10,
            random_state=42
        )
        X_reduced = mapper.fit_transform(data)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom',
            metric='euclidean',
            gen_min_span_tree=True,
        )
        clusterer.fit(X_reduced)

        dbcv_score = clusterer.relative_validity_

        if dbcv_score is None or np.isnan(dbcv_score):
             return 1.0

        return -dbcv_score

    except Exception as e:
        print(f"Error during run: {e}")
        return 1.0

def parameter_optimization(X, n_calls=50, n_random_starts=10):
    search_space = [
        Integer(5, 30, name='min_cluster_size'),
        Integer(5, 30, name='min_samples'),

        Integer(5, 50, name='n_neighbors'),
        #Real(0.0, 0.5, name='min_dist'),
        #Integer(5, 15, name='n_components'),
    ]

    @use_named_args(search_space)
    def fitness(**params):
        if params['min_cluster_size'] > params['min_samples']:
            return 100.0
        return hdbscan_dbcv_score(
            X,
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            n_neighbors=params['n_neighbors'],
            #min_dist=0.0,
            #n_components=params['n_components'],
        )

    results = gp_minimize(
        func=fitness,
        dimensions=search_space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=42,
        verbose=True
    )

    best_neg_dbcv = results.fun
    best_dbcv = -best_neg_dbcv
    best_params = results.x
    best_params_dict = {
        'min_cluster_size': best_params[0],
        'min_samples': best_params[1],
        'n_neighbors': best_params[2],
        #'n_components': best_params[3]
    }

    print(f"Maximum DBCV Score Found: {best_dbcv:.4f}")
    print("Best Parameter Combination:")
    for param, value in best_params_dict.items():
        print(f"  {param:<18}: {value}")

    return best_params_dict


if __name__ == "__main__":

    # ---- Set to True to run Bayesian hyperparameter optimisation,
    #      False to use the default parameters below instead.
    OPTIMIZE = True

    # ---- Default parameters (used when OPTIMIZE = False) -------------------
    DEFAULT_PARAMS = dict(
        n_neighbors      = 37,
        min_cluster_size = 5,
        min_samples      = 10,
    )
    # ------------------------------------------------------------------------

    features = create_feature_vectors()
    features = StandardScaler().fit_transform(features)
    lowd_features = run_PCA(features, ncomp=50)

    if OPTIMIZE:
        print("\n--- Running Bayesian hyperparameter optimisation ---")
        best_params = parameter_optimization(lowd_features)
    else:
        print("\n--- Using default parameters (OPTIMIZE=False) ---")
        best_params = DEFAULT_PARAMS
        for param, value in best_params.items():
            print(f"  {param:<18}: {value}")

    umap_features = run_umap(
        lowd_features,
        n_neighbors  = best_params["n_neighbors"],
        min_dist     = 0.0,
        n_components = 10,
        random_state = 42,
    )

    labels = run_hdbscan(
        umap_features,
        min_cluster_size     = best_params["min_cluster_size"],
        min_samples          = best_params["min_samples"],
        approx_min_span_tree = False,
    )

    plot_umap(lowd_features, labels)
