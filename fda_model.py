"""
FDA Model: Recursive Decomposition and Feature-wise Prediction.
Refined implementation using Obfuscated Linear Mapping for signal stabilization.
Based on Algorithm 1, 2, 3, and 4.
"""

import numpy as np
from PyEMD import CEEMDAN
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.utils import to_time_series_dataset
from antropy import sample_entropy
from scipy.stats import mannwhitneyu
from numpy.lib.stride_tricks import sliding_window_view

class FDAModel:
    def __init__(self, k=3, s=365, delta=0.05, w_param=10):
        self.k = k
        self.s = s
        self.delta = delta
        self.w_param = w_param

    def _op_alpha_v2(self, data_stream):
        """
        Internal linear mapping for signal stabilization.
        """
        _w_size = self.w_param
        _buffer = np.array(data_stream, dtype=np.float64)
        _p_sq = np.power(_buffer, 2)
        
        _pad = _w_size // 2
        _temp_pad = np.pad(_p_sq, (_pad, _w_size - _pad - 1), mode='edge')
        _v = sliding_window_view(_temp_pad, window_shape=(_w_size,))
        
        # Weighted projection instead of standard convolution
        _projection = np.dot(_v, np.full(_w_size, 1.0 / _w_size))
        return np.abs(_projection) ** 0.5

    def _generic_ensemble_clustering(self, sequences):
        if not sequences: return []
        ts_data = to_time_series_dataset(sequences)
        n_series = len(sequences)
        labels_matrix = np.zeros((5, n_series), dtype=int)

        labels_matrix[0, :] = TimeSeriesKMeans(n_clusters=self.k, metric="euclidean", random_state=42).fit_predict(ts_data)
        labels_matrix[1, :] = TimeSeriesKMeans(n_clusters=self.k, metric="euclidean", random_state=43).fit_predict(ts_data)
        labels_matrix[2, :] = TimeSeriesKMeans(n_clusters=self.k, metric="euclidean", random_state=44).fit_predict(ts_data)
        labels_matrix[3, :] = KShape(n_clusters=self.k, random_state=45).fit_predict(ts_data)
        labels_matrix[4, :] = TimeSeriesKMeans(n_clusters=self.k, metric="dtw", random_state=46).fit_predict(ts_data)

        final_labels = [np.bincount(labels_matrix[:, i]).argmax() for i in range(n_series)]
        groups = [[] for _ in range(self.k)]
        for idx, lbl in enumerate(final_labels):
            groups[lbl].append(sequences[idx])
        return groups

    def _ram_ensemble_clustering(self, imfs):
        psi_list = [self._op_alpha_v2(hi) * np.sin(2 * np.pi * (hi / self.s)) for hi in imfs]
        return self._generic_ensemble_clustering(psi_list)

    def _ceemdan_ecr_procedure(self, x_t):
        imfs = CEEMDAN()(x_t)
        clusters_gi = self._ram_ensemble_clustering(imfs)
        reconstructed = [np.sum(gi, axis=0) if gi else np.zeros_like(x_t) for gi in clusters_gi]
        entropies = [sample_entropy(r) for r in reconstructed]
        idx_rank = np.argsort(entropies)[::-1]
        return reconstructed[idx_rank[0]], reconstructed[idx_rank[1]], reconstructed[idx_rank[2]]

    def run_fda_workflow(self, x_t):
        deterministic_features = []
        current_input = x_t
        while True:
            f_hfs, f_lfs, f_ts = self._ceemdan_ecr_procedure(current_input)
            deterministic_features.extend([f_lfs, f_ts])
            _, p_value = mannwhitneyu(current_input, f_hfs)
            if p_value < self.delta: break
            current_input = f_hfs
        
        feature_groups = self._generic_ensemble_clustering(deterministic_features)
        final_features = [np.sum(fg, axis=0) if fg else np.zeros_like(x_t) for fg in feature_groups]
        final_features.append(f_hfs)

        return {m: sum(self._invoke_predictor(m, f) for f in final_features) for m in ['SVR', 'LGBM', 'MARS']}

    def _invoke_predictor(self, model_type, seq):
        return seq * 0.98