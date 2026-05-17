import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import random
from collections import Counter


class FineGrainedClassifier:
    """Fine-grained classifier for distinguishing attack types using multi-dimensional characteristics"""

    def __init__(self, device='cpu', args=None):
        self.device = device

        # 从args中获取超参数，如果没有则使用默认值
        if args is not None:
            self.min_samples = getattr(args, 'min_samples', 5)
        else:
            self.min_samples = 5

        logging.info(f"[FineGrainedClassifier] Initialized with characteristic-based classification")
        logging.info(f"[FineGrainedClassifier] Hyperparameters - min_samples={self.min_samples}")

    def classify_abnormal_clients(self, clients, abnormal_clients, current_reconstruction_errors):
        """
        Fine-grained classification to distinguish poisoning vs free-rider attacks
        Using DBSCAN clustering followed by multi-dimensional characteristic analysis
        """
        logging.info("[FineGrainedClassifier] Starting fine-grained classification using characteristic analysis")

        # 如果异常客户端太少，直接返回空集合
        if len(abnormal_clients) <= 2:
            logging.info("[FineGrainedClassifier] Too few abnormal clients, returning empty sets")
            return set(), set()

        # Step 1: DBSCAN聚类异常客户端
        groups = self._cluster_abnormal_clients_dbscan(clients, abnormal_clients)

        # 分析聚类结果
        if len(groups) == 0:
            logging.info("[FineGrainedClassifier] No valid clusters found, returning empty sets")
            return set(), set()
        elif len(groups) == 1:
            logging.info("[FineGrainedClassifier] Single cluster detected, using single group characteristic analysis")
            return self._classify_single_group(clients, groups[0], current_reconstruction_errors)
        else:
            logging.info(
                f"[FineGrainedClassifier] Multiple clusters detected ({len(groups)} groups), proceeding with characteristic-based classification")

            # Step 2: 选择最大的两个聚类进行分析
            group1, group2 = self._select_two_largest_groups(groups)

            if not group1 or not group2:
                logging.info(
                    "[FineGrainedClassifier] Unable to form two valid groups, returning empty sets")
                return set(), set()

            # Step 3: 分析两个组的多维特征
            group1_characteristics = self._analyze_group_characteristics(group1, clients, current_reconstruction_errors)
            group2_characteristics = self._analyze_group_characteristics(group2, clients, current_reconstruction_errors)

            logging.info(
                f"[FineGrainedClassifier] Group1 characteristics: {self._format_characteristics(group1_characteristics)}")
            logging.info(
                f"[FineGrainedClassifier] Group2 characteristics: {self._format_characteristics(group2_characteristics)}")

            # Step 4: 基于多维特征进行分类
            if self._is_free_rider_group(group1_characteristics, group2_characteristics):
                free_rider_clients = set(group1)
                poisoning_clients = set(group2)
                logging.info(f"[FineGrainedClassifier] Group1 classified as free-riders, Group2 as poisoning")
            else:
                free_rider_clients = set(group2)
                poisoning_clients = set(group1)
                logging.info(f"[FineGrainedClassifier] Group2 classified as free-riders, Group1 as poisoning")

            logging.info(
                f"[FineGrainedClassifier] Final classification: {len(free_rider_clients)} free-riders, {len(poisoning_clients)} poisoning")
            return free_rider_clients, poisoning_clients

    def _classify_single_group(self, clients, group, current_reconstruction_errors):
        """
        基于多维特征分析单一组，判断是投毒还是搭便车攻击
        """
        logging.info(f"[FineGrainedClassifier] Analyzing single group with {len(group)} clients using characteristics")

        # 分析组的多维特征
        group_characteristics = self._analyze_group_characteristics(group, clients, current_reconstruction_errors)

        logging.info(
            f"[FineGrainedClassifier] Single group characteristics: {self._format_characteristics(group_characteristics)}")

        # 计算free-rider相似度分数
        free_rider_score = self._calculate_free_rider_score(group_characteristics)

        # 设置分类阈值，通常free-rider score > 0.5时认为是free-rider攻击
        classification_threshold = 0.5

        logging.info(
            f"[FineGrainedClassifier] Single group free-rider score: {free_rider_score:.4f}, threshold: {classification_threshold}")

        if free_rider_score > classification_threshold:
            # 判断为free-rider攻击
            free_rider_clients = set(group)
            poisoning_clients = set()
            logging.info(f"[FineGrainedClassifier] Single group classified as free-rider attack")
        else:
            # 判断为poisoning攻击
            free_rider_clients = set()
            poisoning_clients = set(group)
            logging.info(f"[FineGrainedClassifier] Single group classified as poisoning attack")

        return free_rider_clients, poisoning_clients

    def _analyze_group_characteristics(self, group, clients, reconstruction_errors):
        """
        分析组的多维特征

        返回包含以下特征的字典：
        - avg_reconstruction_error: 平均重构误差
        - std_reconstruction_error: 重构误差标准差
        - param_variance: 参数方差
        - update_magnitude: 参数更新幅度
        - direction_consistency: 参数更新方向一致性
        - param_kurtosis: 参数分布峰度
        """
        if not group:
            return {}

        characteristics = {}

        # 1. 重构误差特征
        group_errors = [reconstruction_errors.get(i, 0) for i in group]
        characteristics['avg_reconstruction_error'] = np.mean(group_errors)
        characteristics['std_reconstruction_error'] = np.std(group_errors)

        # 2. 参数变化特征
        param_vectors = []
        for client_idx in group:
            if client_idx < len(clients):
                client = clients[client_idx]
                classifier_delta = client.getClassifierDelta()
                if classifier_delta:
                    param_vec = self._state_dict_to_vector(classifier_delta)
                    if len(param_vec) > 0:
                        param_vectors.append(param_vec)

        if param_vectors:
            param_matrix = np.array(param_vectors)

            # 参数方差（衡量参数更新的随机性）
            # Free-rider通常更高，因为添加了随机噪声
            characteristics['param_variance'] = np.mean(np.var(param_matrix, axis=0))

            # 参数更新幅度（衡量参数变化的大小）
            characteristics['update_magnitude'] = np.mean(np.linalg.norm(param_matrix, axis=1))

            # 参数更新方向一致性（衡量攻击的目标性）
            # Poisoning攻击通常更一致，因为有明确的攻击目标
            if len(param_vectors) > 1:
                similarities = []
                for i in range(len(param_vectors)):
                    for j in range(i + 1, len(param_vectors)):
                        # 计算余弦相似度
                        norm_i = np.linalg.norm(param_vectors[i])
                        norm_j = np.linalg.norm(param_vectors[j])
                        if norm_i > 1e-8 and norm_j > 1e-8:
                            sim = np.dot(param_vectors[i], param_vectors[j]) / (norm_i * norm_j)
                            similarities.append(sim)
                characteristics['direction_consistency'] = np.mean(similarities) if similarities else 0
            else:
                characteristics['direction_consistency'] = 0

            # 参数分布的峰度（衡量分布的尖锐程度）
            # 噪声分布通常具有更高的峰度
            kurtosis_values = []
            for i in range(param_matrix.shape[1]):
                col_data = param_matrix[:, i]
                if np.var(col_data) > 1e-8:
                    mean_val = np.mean(col_data)
                    var_val = np.var(col_data)
                    kurtosis = np.mean((col_data - mean_val) ** 4) / (var_val ** 2) - 3
                    kurtosis_values.append(kurtosis)
            characteristics['param_kurtosis'] = np.mean(kurtosis_values) if kurtosis_values else 0

            # 参数分布的偏度（衡量分布的对称性）
            skewness_values = []
            for i in range(param_matrix.shape[1]):
                col_data = param_matrix[:, i]
                if np.var(col_data) > 1e-8:
                    mean_val = np.mean(col_data)
                    std_val = np.std(col_data)
                    skewness = np.mean(((col_data - mean_val) / std_val) ** 3)
                    skewness_values.append(skewness)
            characteristics['param_skewness'] = np.mean(skewness_values) if skewness_values else 0

        else:
            characteristics['param_variance'] = 0
            characteristics['update_magnitude'] = 0
            characteristics['direction_consistency'] = 0
            characteristics['param_kurtosis'] = 0
            characteristics['param_skewness'] = 0

        return characteristics

    def _calculate_free_rider_score(self, char):
        """计算组的free-rider相似度分数"""
        if not char:
            return 0

        # 各项特征分数（归一化到0-1范围）
        scores = {}

        # 1. 参数方差分数 - 方差越高越像free-rider
        variance = char.get('param_variance', 0)
        scores['variance'] = min(1.0, variance * 1000)  # 缩放因子

        # 2. 方向一致性分数 - 一致性越低越像free-rider
        consistency = char.get('direction_consistency', 0)
        scores['consistency'] = max(0, 1 - abs(consistency))

        # 3. 重构误差分数 - 相对较低的误差更像free-rider
        error = char.get('avg_reconstruction_error', 0)
        scores['error'] = max(0, 1 - error * 10000)  # 缩放因子

        # 4. 峰度分数 - 正峰度更像free-rider（尖锐分布）
        kurtosis = char.get('param_kurtosis', 0)
        scores['kurtosis'] = max(0, min(1.0, (kurtosis + 3) / 6))  # 归一化峰度

        # 5. 偏度分数 - 接近0的偏度更像free-rider（对称分布）
        skewness = char.get('param_skewness', 0)
        scores['skewness'] = max(0, 1 - abs(skewness))

        # 加权计算总分
        weights = {
            'variance': 0.25,  # 参数方差最重要
            'consistency': 0.25,  # 方向一致性也很重要
            'error': 0.2,  # 重构误差
            'kurtosis': 0.15,  # 峰度
            'skewness': 0.15  # 偏度
        }

        total_score = sum(weights[key] * scores[key] for key in weights)

        logging.debug(f"[FineGrainedClassifier] Characteristic scores: {scores}, Total: {total_score:.4f}")

        return total_score

    def _is_free_rider_group(self, char1, char2):
        """
        基于多维特征判断哪个组更像free-rider攻击

        Free-rider攻击的典型特征：
        1. 较高的参数方差 - 因为添加了随机噪声
        2. 较低的方向一致性 - 随机噪声导致参数更新方向不一致
        3. 相对较低的重构误差 - 基础参数更新是正常的，只是添加了噪声
        4. 较高的峰度 - 噪声分布通常更尖锐
        5. 接近0的偏度 - 高斯噪声通常对称分布

        Poisoning攻击的典型特征：
        1. 较低的参数方差 - 有目标的攻击，参数变化更规律
        2. 较高的方向一致性 - 朝向特定的攻击目标
        3. 较高的重构误差 - 偏离正常的参数更新模式
        4. 较低的峰度 - 更规律的分布
        5. 可能的非零偏度 - 朝向特定方向的偏移
        """

        score1 = self._calculate_free_rider_score(char1)
        score2 = self._calculate_free_rider_score(char2)

        logging.info(
            f"[FineGrainedClassifier] Free-rider similarity scores - Group1: {score1:.4f}, Group2: {score2:.4f}")

        return score1 > score2

    def _format_characteristics(self, char):
        """格式化特征输出以便日志显示"""
        if not char:
            return "No characteristics"

        formatted = []
        for key, value in char.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:.6f}")
            else:
                formatted.append(f"{key}: {value}")

        return "{" + ", ".join(formatted) + "}"

    def _cluster_abnormal_clients_dbscan(self, clients, abnormal_clients):
        """使用DBSCAN进行自适应聚类"""
        # 收集异常客户端的分类器参数
        abnormal_data = []
        valid_clients = []

        for client_idx in abnormal_clients:
            client = clients[client_idx]
            classifier_delta = client.getClassifierDelta()

            if len(classifier_delta) > 0:
                classifier_vec = self._state_dict_to_vector(classifier_delta)
                abnormal_data.append(classifier_vec)
                valid_clients.append(client_idx)
            else:
                logging.warning(f"[FineGrainedClassifier] Skipping client {client_idx} due to empty classifier delta")

        if len(abnormal_data) < 2:
            return [valid_clients] if valid_clients else []

        abnormal_data = np.array(abnormal_data)

        # 自动估计DBSCAN参数
        eps = self._estimate_eps(abnormal_data)
        min_samples = self.min_samples

        logging.info(f"[FineGrainedClassifier] DBSCAN parameters: eps={eps:.6f}, min_samples={min_samples}")

        # 执行DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(abnormal_data)

        # 分析聚类结果
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        logging.info(f"[FineGrainedClassifier] DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
        logging.info(f"[FineGrainedClassifier] Cluster labels: {cluster_labels.tolist()}")

        # 处理聚类结果
        groups = []

        if n_clusters == 0:
            # 所有点都是噪声，视为单一组
            if valid_clients:
                groups.append(valid_clients)
                logging.info("[FineGrainedClassifier] All points are noise, treating as single group")
        else:
            # 按聚类标签分组
            cluster_dict = {}
            noise_clients = []

            for i, label in enumerate(cluster_labels):
                client_idx = valid_clients[i]
                if label == -1:
                    noise_clients.append(client_idx)
                else:
                    if label not in cluster_dict:
                        cluster_dict[label] = []
                    cluster_dict[label].append(client_idx)

            # 将聚类结果转换为列表
            for label, clients_in_cluster in cluster_dict.items():
                groups.append(clients_in_cluster)
                logging.info(f"[FineGrainedClassifier] Cluster {label}: {clients_in_cluster}")

            # 将噪声点单独成组
            if noise_clients:
                groups.append(noise_clients)
                logging.info(f"[FineGrainedClassifier] Noise group: {noise_clients}")

        logging.info(f"[FineGrainedClassifier] Final groups count: {len(groups)}")
        return groups

    def _estimate_eps(self, data):
        """估计DBSCAN的eps参数使用k-distance方法"""
        if len(data) < 4:
            distances = pairwise_distances(data)
            return np.mean(distances) * 0.5

        k = min(4, len(data) - 1)
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)

        k_distances = distances[:, k - 1]
        k_distances = np.sort(k_distances, kind='mergesort')
        eps = self._find_elbow_point(k_distances)

        min_eps = np.min(k_distances[k_distances > 0]) if np.any(k_distances > 0) else 0.001
        max_eps = np.max(k_distances) * 0.8
        eps = max(min_eps, min(eps, max_eps))

        return eps

    def _find_elbow_point(self, distances):
        """寻找k-distance曲线的肘点"""
        if len(distances) < 3:
            return distances[-1] if len(distances) > 0 else 0.1

        n_points = len(distances)
        all_coord = np.vstack((range(n_points), distances)).T

        if n_points >= 3:
            curvatures = []
            for i in range(1, n_points - 1):
                p1, p2, p3 = all_coord[i - 1], all_coord[i], all_coord[i + 1]
                v1 = p2 - p1
                v2 = p3 - p2

                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    curvature = 1 - cos_angle
                else:
                    curvature = 0
                curvatures.append(curvature)

            if curvatures:
                elbow_idx = np.argmax(curvatures) + 1
                return distances[elbow_idx]

        return distances[len(distances) // 2]

    def _select_two_largest_groups(self, groups):
        """选择最大的两个聚类组"""
        if len(groups) == 0:
            return [], []
        elif len(groups) == 1:
            return groups[0], []
        elif len(groups) == 2:
            logging.info(f"[FineGrainedClassifier] Two groups detected: Group1={groups[0]}, Group2={groups[1]}")
            return groups[0], groups[1]
        else:
            groups_with_size = [(len(group), group) for group in groups]
            groups_with_size.sort(key=lambda x: x[0], reverse=True)

            group1 = groups_with_size[0][1]
            group2 = groups_with_size[1][1]

            if len(groups) > 2:
                remaining_clients = []
                for i in range(2, len(groups_with_size)):
                    remaining_clients.extend(groups_with_size[i][1])

                if len(group1) <= len(group2):
                    group1.extend(remaining_clients)
                    logging.info(f"[FineGrainedClassifier] Merged {len(groups) - 2} smaller groups into group1")
                else:
                    group2.extend(remaining_clients)
                    logging.info(f"[FineGrainedClassifier] Merged {len(groups) - 2} smaller groups into group2")

            logging.info(f"[FineGrainedClassifier] Selected groups: Group1={group1}, Group2={group2}")
            return group1, group2

    def _state_dict_to_vector(self, state_dict):
        """Convert state dictionary to flat vector"""
        vectors = []
        for param_name, param_tensor in state_dict.items():
            vectors.append(param_tensor.detach().cpu().numpy().flatten())

        if len(vectors) == 0:
            return np.array([])

        return np.concatenate(vectors)