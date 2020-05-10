#!/usr/bin/env python
# coding: utf-8

"""Module containing classes and functions for performing rest2vec.

Description:
    This module contains the PhaseEmbedding object for the equations
    described in [1]. In addition to the classes, the individual methods
    have been included as internal functions (preceded by '_').

Reference:
    [1] Morrissey et al. (2020). rest2vec: A kernel method for studying the
        intrinsic geometry of functional connectomes.

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


class PhaseEmbedding:
    """Class for creating phase angle spatial embedding (PhASE) objects."""
    def __init__(self, nw, roi_xyz, roi_names):
        self.nw = nw
        self.roi_names = roi_names
        self.roi_xyz = roi_xyz

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

            P = np.sum(self.nw < 0, axis=2) / num_subjects
            return P

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

            return np.arctan(np.sqrt(self.P / (1 - self.P)))

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
                K = np.zeros_like(self.theta)

                for i in range(N):
                    row_i = self.theta[i]
                    K[i, :] = np.sum(np.cos(row_i - self.theta), axis=1) / N

                return K

            elif method == 'rbf':
                K = np.ones_like(self.theta)

                rows, cols = np.triu_indices(N)

                for i in rows:
                    row_i = self.theta[i]
                    for j in cols:
                        row_j = self.theta[j, :]
                        K[i, j] = np.exp(-sigma *
                                         np.sum((row_i - row_j)**2) / N)

                K = np.fill_diagonal(K + K.T, 1)

                return K

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
            Kcen = C @ self.K @ C  # center kernel matrix

            # Get eigenvector corresponding to maximum eigenvalue
            evals, evecs = np.linalg.eig(Kcen)
            qmax = evecs[:, np.argmax(np.diag(evals))]

            # Create community assignments
            mod = qmax.copy()
            mod[mod >= threshold] = 0
            mod[mod < threshold] = 1


            return mod, qmax, Kcen

        self.avg_nw = np.mean(nw, axis=2)
        self.P = npm(self)
        self.theta = phase(self)
        self.K = kernel_sim(self)
        self.mod, self.qmax, self.Kcen = modularity(self)

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


"""Standalone PhaseEmbedding method functions."""


def _npm(group_matrix):
    """Compute probability of negative correlaiton matrix.

    Parameters
    ----------
    group_matrix : numpy array
        3D array of size N x N x Z, where N is the number of ROIs, and Z is
        the number of subjects. Each N x N slice is the Pearson correlation
        connectome for the z-th subject.

    Returns
    -------
    npm : numpy array
        N x N array where the (i,j)-th elements are the probability of there
        being a negative edge between nodes i and j.
    """

    num_subjects = group_matrix.shape[2]

    return np.sum(group_matrix < 0, axis=2) / num_subjects


def _phase(npm):
    """Compute phase angle spatial embedding (PhASE) matrix from negative
    probability matrix.

    Parameters
    ----------
    npm : numpy array
        N x N array where the (i,j)-th elements are the probability of there
        being a negative edge between nodes i and j.

    Returns
    -------
    theta : numpy array
        N x N array where the (i,j)-th elements are the phase angle between
        i and j.
    """

    return np.arctan(np.sqrt(npm / (1 - npm)))


def _kernel_sim(theta, method='cosine', sigma=1):
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

    N = theta.shape[0]  # number of ROIs

    if method == 'cosine':
        K = np.zeros_like(theta)

        for i in range(N):
            row_i = theta[i]
            K[i, :] = np.sum(np.cos(row_i - theta), axis=1) / N

        return K

    elif method == 'rbf':
        K = np.ones_like(theta)

        rows, cols = np.triu_indices(N)

        for i in rows:
            row_i = theta[i]
            for j in cols:
                row_j = theta[j, :]
                K[i, j] = np.exp(-sigma * np.sum((row_i - row_j)**2) / N)

        K = np.fill_diagonal(K + K.T, 1)

        return K


def _find_modularity(K, threshold=0):
    """Compute modularity of phase angle spatial embedding (PhASE) matrix
    after computing kernel similarity matrix.

    Parameters
    ----------
    K : numpy array
        N x N kernel similarity matrix. See compute_k() for details.

    threshold : float
        Value at which to threshold top eigenvector to determine cut.
        Default is 0.

    Returns
    -------
    mod : N x 1 community assignment vector.
    """

    N = K.shape[0]  # number ROIs
    C = np.eye(N) - (1 / N) * np.ones_like(N)  # centering matrix
    Kcen = C @ K @ C  # center kernel matrix

    # Get eigenvector corresponding to maximum eigenvalue
    evals, evecs = np.linalg.eig(Kcen)
    qmax = evecs[:, np.argmax(np.diag(evals))]

    # Create community assignments
    mod = qmax.copy()
    mod[mod >= threshold] = 1
    mod[mod < threshold] = 2

    return mod
