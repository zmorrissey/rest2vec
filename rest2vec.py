#!/usr/bin/env python
# coding: utf-8

"""Module containing classes and functions for performing rest2vec.

Description:
    This module contains the PhaseEmbedding object for the equations
    described in [1]. In addition to the classes, the individual methods
    have been included as internal functions (preceded by '_').

Reference:
    [1] Morrissey et al. (2020). rest2vec: Vectorizing the resting-state
        functional connectome using graph embedding.
        DOI: https://doi.org/10.1101/2020.05.10.085332
        URL: https://www.biorxiv.org/content/early/2020/05/12/2020.05.10.085332

Contact:
    First author
    ------------
    Zachery D. Morrissey (B.S.)
    Ph.D. Candidate
    Department of Psychiatry
    Graduate Program in Neuroscience
    University of Illinois at Chicago
    zmorri4@uic.edu

    Senior/corresponding author
    ---------------------------
    Alex D. Leow (M.D., Ph.D.)
    Associate Professor
    Departments of Psychiatry, Bioengineering, and Computer Science
    University of Illinois at Chicago
    alexfeuillet@gmail.com
"""

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics.pairwise import euclidean_distances as edist
from nibabel.affines import apply_affine
from nilearn import datasets
from nilearn.plotting import plot_connectome
from sklearn.manifold import Isomap


def coord_polar(mat):
    """Convert 2D array from Cartesian (x, y) coordinates to
    polar (r, theta) coordinates.

    Parameters
    ----------
    mat : numpy array
        N x 2 array of Cartesian coordinates.

    Returns
    -------
    r : numpy vector
        Polar radius vector of size N.

    theta : numpy vector
        Polar angle vector of size N.
    """
    x = mat[:, 0].copy()
    y = mat[:, 1].copy()

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return r, theta


class PhaseEmbedding:
    """Class for creating phase angle spatial embedding (PhASE) objects."""
    def __init__(self, nw, roi_xyz, roi_names, n_neighbors):
        self.nw = nw
        self.roi_names = roi_names
        self.roi_xyz = roi_xyz
        self.n_neighbors = n_neighbors  # for isomap

    def npm(self):
        """Compute probability of negative correlation matrix.

        Parameters
        ----------
        nw : numpy array
            3D array of size N x N x Z, where N is the number of ROIs,
            and Z is the number of subjects. Each N x N slice is the
            Pearson correlation connectome for the z-th subject.

        Returns
        -------
        P : numpy array
            N x N array where the (i,j)-th elements are the probability
            of there being a negative edge between nodes i and j.
        """

        num_subjects = self.nw.shape[2]

        self.P = np.sum(self.nw < 0, axis=2) / num_subjects
        return self

    def phase(self):
        """Compute phase angle spatial embedding (PhASE) matrix from
        negative probability matrix.

        Parameters
        ----------
        npm : numpy array
            N x N array where the (i,j)-th elements are the probability
            of there being a negative edge between nodes i and j.

        Returns
        -------
        theta : numpy array
            N x N array where the (i,j)-th elements are the phase angle
            between i and j.
        """

        self.theta = np.arctan(np.sqrt(self.P / (1 - self.P)))
        return self

    def kernel_sim(self, method='cosine', sigma=1):
        """Compute kernel similarity matrix.

        Parameters
        ----------
        theta : numpy array
            N x N phase angle spatial embedding matrix. See phase() for
            details.

        method : string
            Kernel method to use (cosine or RBF). Default is cosine.

        sigma : float
            If using RBF kernel, value of sigma to use.

        Returns
        -------
        K : numpy array
            N x N kernel similarity matrix.
        """

        N = self.theta.shape[0]  # number of ROIs

        if method == 'cosine':
            self.K = np.zeros_like(self.theta)

            for i in range(N):
                row_i = self.theta[i]
                self.K[i, :] = np.sum(np.cos(row_i - self.theta), axis=1) / N

            return self

        elif method == 'rbf':
            self.K = np.ones_like(self.theta)

            rows, cols = np.triu_indices(N)

            for i in rows:
                row_i = self.theta[i]
                for j in cols:
                    row_j = self.theta[j, :]
                    self.K[i, j] = np.exp(-sigma *
                                          np.sum((row_i - row_j)**2) / N)

            self.K = np.fill_diagonal(self.K + self.K.T, 1)

            return self

    def modularity(self, threshold=0):
        """Compute modularity of phase angle spatial embedding (PhASE)
           matrix after computing kernel similarity matrix.

        Parameters
        ----------
        K : numpy array
            N x N kernel similarity matrix. See compute_k() for details.

        threshold : float
            Value at which to threshold top eigenvector to determine cut.
            Default is 0.

        Returns
        -------
        mod : numpy array
            N x 1 community assignment vector.

        qmax : numpy array
            Eigenvector corresponding to maximum eigenvalue
        """

        N = self.K.shape[0]  # number ROIs
        C = np.eye(N) - (1 / N) * np.ones_like(N)  # centering matrix
        self.Kcen = C @ self.K @ C  # center kernel matrix

        # Get eigenvector corresponding to maximum eigenvalue
        self.evals, self.evecs = np.linalg.eig(self.Kcen)
        self.qmax = self.evecs[:, np.argmax(np.diag(self.evals))]

        # Create community assignments
        self.mod = self.qmax.copy()
        self.mod[self.mod >= threshold] = 0
        self.mod[self.mod < threshold] = 1

        return self

    def polar_embedding(self):
        """Compute polar coordinates for embedding."""
        self.isomap_r, self.isomap_theta = coord_polar(self.isomap)

        return self

    def distance_to_origin(self):
        """Compute distance to origin of embedding."""

        self.D = edist(self.isomap,
                       np.zeros([1, self.isomap.shape[1]])).flatten()

    def fit(self):
        """Main call to run rest2vec."""
        self.avg_nw = np.mean(self.nw, axis=2)
        self.npm()
        self.phase()
        self.kernel_sim()
        self.modularity()

        # Dimensionality reduction
        self.isomap = Isomap(
            n_neighbors=self.n_neighbors,
            n_components=2,
            path_method='D').fit_transform(self.theta)

        self.polar_embedding()
        self.distance_to_origin()

        # Build embedding dataframe
        self.df = pd.DataFrame({
            'x': self.isomap[:, 0],
            'y': self.isomap[:, 1],
            'theta': self.isomap_theta,
            'r': self.isomap_r,
            'D': self.D,
            'mod': self.mod.astype(int),
            'mni_x': self.roi_xyz['x'],
            'mni_y': self.roi_xyz['y'],
            'mni_z': self.roi_xyz['z'],
            'roi': self.roi_names.replace('_', ' ', regex=True)})

        return self

    def to_brainnet(self, edges=None, C=None, S=None,
                    path='.', prefix='r2v-brainnet'):
        """Export data to plaintext file(s) for use with BrainNet Viewer
        [1]. For details regarding .node and .edge file construction, the
        user is directed to the BrainNet Viewer User Manual.

        This code was quality tested using BrainNet version 1.61 released on
        2017-10-31 with MATLAB 9.3.0.713579 (R2017b).

        Parameters:
        -----------
        edges : numpy array
            N x N matrix containing edge values (default is avg_nw).

        roi_xyz : pandas dataframe
            N x 3 dataframe containing the (x, y, z) MNI coordinates of each
            brain ROI.

        S : pandas series
            Node size value (defaults to same size).

        C : pandas series
            Node color value (defaults to same color). For modular color,
            use integers; for continuous data use floats.

        roi_names : pandas series
            Names of each ROI as string.

        path : string
            Path to output directory (default is current directory). Note:
            do not include trailing '/' at end.

        prefix : string
            Filename prefix for output files.

        Returns
        -------
        <prefix>.node, <prefix>.edge : files
            Plaintext output files for input to BrainNet.

        References
        ----------
        [1] Xia M, Wang J, He Y (2013) BrainNet Viewer: A Network
            Visualization Tool for Human Brain Connectomics. PLoS ONE 8:
            e68910.
        """

        N = len(self.roi_xyz)  # number of nodes

        if edges is None:
            edges = self.avg_nw

        if C is None:
            C = np.ones(N)

        if S is None:
            S = np.ones(N)

        # BrainNet does not recognize node labels with white space, replace
        # spaces with underscore
        names = self.roi_names.str.replace(' ', '_')

        # Build .node dataframe
        df = self.roi_xyz.copy()
        df = (df
              .assign(C=C)
              .assign(S=S)
              .assign(names=names))

        # Output .node file
        df.to_csv(f'{path}/{prefix}.node', sep='\t',
                  header=False, index=False)
        print(f'Saved {path}/{prefix}.node.')

        # Output .edge file
        np.savetxt(f'{path}/{prefix}.edge', edges, delimiter='\t')
        print(f'Saved {path}/{prefix}.edge')


class Neurosynth:
    """Class to store and plot data from Neurosynth."""
    def __init__(self, nsynth_path, embedding):
        self.embedding = embedding
        self.mni_xyz = embedding.roi_xyz.to_numpy()
        self.nsynth_nii = nib.load(nsynth_path)
        self.nsynth_mat = self.nsynth_nii.get_data()

        def nsynth_vals_from_mni(self):
            """Get the data values at MNI_XYZ from a neurosynth map
            registered to the MNI template.

            Parameters
            ----------
            self.mni_xyz : numpy array
                MNI (x,y,z)-coordinates to look up in neurosynth map.

            self.nsynth_map : nibabel nifti object
                Neurosynth nifti file (loaded using nibabel.load)

            Returns
            -------
            nsynth_vals : numpy array
                Nx1 array of values at each MNI coordinate from
                neurosynth data map
            """
            nsynth_vals = np.zeros(self.mni_xyz.shape[0], dtype=int)
            mni152 = datasets.load_mni152_template()

            for i in range(self.mni_xyz.shape[0]):
                x, y, z = [int(np.round(i))
                           for i in
                           apply_affine(np.linalg.inv(mni152.affine),
                                        [self.mni_xyz[i, 0],
                                         self.mni_xyz[i, 1],
                                         self.mni_xyz[i, 2]])]
                nsynth_vals[i] = self.nsynth_mat[x, y, z]

            return nsynth_vals

        self.nsynth_vals = nsynth_vals_from_mni(self)

    def plot_nsynth_vals(self, df, ax, **kwargs):
        """Plot polar embedding plot where Neurosynth data
        values are mapped to a color gradient.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe containing the 'r' and 'theta' coordinates
            to plot each ROI.

        ax : matplotlib axes
            Axes to plot to.

        kwargs : dict
            Optional keyword arguments to pass to plt.scatter().
        """
        val_idx = np.where(self.nsynth_vals > 0)

        # Background regions
        ax.scatter(df['theta'].loc[~df.index.isin(val_idx)],
                   df['r'].loc[~df.index.isin(val_idx)],
                   color=[0.7, 0.7, 0.7],
                   alpha=0.25,
                   **kwargs)

        # Foreground -- regions with FDR z-score > 0
        ax.scatter(df['theta'].iloc[val_idx],
                   df['r'].iloc[val_idx],
                   vmin=0,
                   c=self.nsynth_vals[val_idx],
                   **kwargs)

        return ax

    def plot_nsynth_brain(self, ax, **kwargs):
        """Plot glass brain where Neurosynth data
        values are mapped to a color gradient.

        Parameters
        ----------
        nw : numpy array
            N x N Pearson correlation connectome.

        xyz : numpy array
            N x 3 MNI (x, y, z)-coordinates.

        vals : numpy array
            Data values from Neurosynth to map to color.

        ax : matplotlib axes
            Axes to plot to.

        kwargs : dict
            Optional keyword arguments to pass to
            nilearn.plot_connectome().
        """

        val_idx = np.where(self.nsynth_vals > 0)

        plot_connectome(self.embedding.avg_nw[np.ix_(val_idx[0],
                                                    val_idx[0])],
                        node_coords=self.mni_xyz[val_idx],
                        node_color=self.nsynth_vals[val_idx],
                        colorbar=False,
                        edge_threshold=1,
                        annotate=False,
                        node_kwargs={'cmap': 'magma', 'vmin': 0},
                        axes=ax,
                        **kwargs)

        return ax
