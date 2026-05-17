import numpy as np
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading


class ContinuousPOMDP:
    """Continuous State POMDP for long-term anomaly detection"""

    def __init__(self, num_clients, device='cpu'):
        self.num_clients = num_clients
        self.device = device

        # POMDP parameters
        self.gamma = 0.95
        self.Ca = 0.8 #1.0 0.5
        self.CL = 0.3 #0.5 0.8
        self.CR = 0.1 #0.3 0.2

        # 减少计算复杂度但保持理论完整性
        self.Theta = 5  # 傅里叶谐波数量
        self.fourier_coeffs_c = np.random.normal(0, 0.1, self.Theta)
        self.fourier_coeffs_d = np.random.normal(0, 0.1, self.Theta)

        # 粒子滤波参数
        self.num_particles = 30  # 适度减少
        self.particles = np.random.uniform(0, 1, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.mu = 1.5

        # 遗传算法参数（减少但保持理论结构）
        self.population_size = 20  # 从50减少到20
        self.max_generations = 30  # 从100减少到30
        self.M = 40  # 从100减少到40
        self.k = 12  # 从20减少到12

        # 交叉参数（方程35）
        self.pmin = 0.1
        self.pmax = 0.9

        # LPM变异参数（方程38-42）
        self.beta1 = 0.1
        self.beta2 = 0.1
        self.beta3 = 0.1
        self.tau_upper_max = 5.0  # 适度减少
        self.tau_upper_unit = 0.1
        self.g1 = 8  # 适度减少
        self.g2 = 15  # 适度减少
        self.alpha1 = 0.01
        self.beta4 = 0.1
        self.beta5 = 0.05

        # LPM状态变量
        self.mutation_factors = {}
        self.tau_g = 1.0

        # 状态和观察跟踪
        self.current_belief_state = np.ones(self.M) / self.M
        self.state_history = []
        self.observation_history = []

        # 并行化参数
        self.use_parallel_fitness = True
        self.num_workers = min(2, mp.cpu_count())

        # 优化参数
        self.convergence_threshold = 1e-4
        self.convergence_patience = 3

        logging.info(f"[POMDP] Initialized with {num_clients} clients (theory-compliant optimization)")

    def fourier_belief_approximation(self, particles, weights):
        """符合理论的傅里叶信念近似"""
        weights = weights / (np.sum(weights) + 1e-8)

        # 向量化计算傅里叶系数
        for i in range(self.Theta):
            angle = 2 * np.pi * i * particles
            self.fourier_coeffs_c[i] = np.sum(weights * np.sin(angle))
            self.fourier_coeffs_d[i] = np.sum(weights * np.cos(angle))

        # 归一化以满足∫b(s)ds = 1
        total_weight = np.sum(np.abs(self.fourier_coeffs_c)) + np.sum(np.abs(self.fourier_coeffs_d))
        if total_weight > 1e-8:
            self.fourier_coeffs_c /= total_weight
            self.fourier_coeffs_d /= total_weight

    def compute_ess(self, weights):
        """计算有效样本大小（方程22）"""
        return 1.0 / (np.sum(weights ** 2) + 1e-8)

    def kld_sampling(self, k, zeta=0.01, eta=0.01):
        """KLD采样（方程23）"""
        z_quantile = 1.96
        if k <= 1:
            return max(10, self.num_particles // 2)

        numerator = k - 1
        denominator = 2 * zeta
        term1 = 1 - 2 / (9 * (k - 1))
        term2 = np.sqrt(2 / (9 * (k - 1))) * z_quantile
        N = numerator / denominator * (term1 + term2) ** 3
        return max(int(N), 10)

    def adaptive_particle_filtering(self, observation, action):
        """自适应粒子滤波（方程20-26）"""
        # 预测步骤
        noise_std = 0.05
        for i in range(len(self.particles)):
            noise = np.random.normal(0, noise_std)
            if action == 1:  # check action
                self.particles[i] = max(0, min(1, self.particles[i] * 0.8 + noise))
            else:  # monitor action
                self.particles[i] = max(0, min(1, self.particles[i] * 1.1 + noise))

        # 贝叶斯更新权重（方程25）
        for i in range(len(self.particles)):
            obs_prob = np.exp(-0.5 * (observation - self.particles[i]) ** 2 / 0.01)
            self.particles[i] = obs_prob

        # 归一化权重
        total_weight = np.sum(self.weights)
        if total_weight > 1e-8:
            self.weights /= total_weight
        else:
            self.weights = np.ones(len(self.particles)) / len(self.particles)

        # 检查是否需要重采样
        ess = self.compute_ess(self.weights)
        if len(self.particles) / ess > self.mu:
            # 使用KLD采样重采样
            new_size = self.kld_sampling(len(np.unique(self.particles)))
            if new_size != len(self.particles):
                indices = np.random.choice(len(self.particles), size=new_size, p=self.weights)
                self.particles = self.particles[indices]
                self.weights = np.ones(len(self.particles)) / len(self.particles)

        # 更新傅里叶信念近似（方程26）
        self.fourier_belief_approximation(self.particles, self.weights)

    def optimized_evaluate_fitness(self, chromosome, observation):
        """优化的适应度评估，但保持理论结构"""
        if np.sum(chromosome) == 0 or np.sum(chromosome) > self.k:
            return -np.inf

        selected_states = np.where(chromosome == 1)[0] / self.M
        total_reward = 0

        for state in selected_states:
            # 使用傅里叶系数计算状态概率
            state_prob = 0
            for i in range(self.Theta):
                state_prob += self.fourier_coeffs_c[i] * np.sin(2 * np.pi * i * state)
                state_prob += self.fourier_coeffs_d[i] * np.cos(2 * np.pi * i * state)
            state_prob = max(0, state_prob)

            # 计算两种动作的奖励（方程18-19）
            reward_monitor = -self.Ca * state
            reward_check = -self.CL - self.CR * state

            max_reward = max(reward_monitor, reward_check)
            total_reward += state_prob * max_reward

        # 大小惩罚
        size_penalty = -0.01 * np.sum(chromosome)
        return total_reward + size_penalty

    def parallel_evaluate_fitness(self, population, observation):
        """并行适应度评估"""
        if not self.use_parallel_fitness or len(population) < 4:
            return [self.optimized_evaluate_fitness(chromosome, observation) for chromosome in population]

        try:
            fitness_args = [(chromosome, observation, self._get_fitness_params())
                            for chromosome in population]

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                fitness_scores = list(executor.map(self._evaluate_fitness_worker, fitness_args))
            return fitness_scores
        except Exception as e:
            logging.warning(f"[POMDP] Parallel fitness evaluation failed: {e}")
            return [self.optimized_evaluate_fitness(chromosome, observation) for chromosome in population]

    def evolutionary_state_subset_optimization(self, observation):
        """遗传算法状态子集优化（符合论文理论）"""
        # 种群初始化（符合3.3节描述）
        population = []
        for u in range(self.population_size):
            chromosome = np.zeros(self.M, dtype=int)
            # 稀疏性控制初始化（ς = 0.3）
            k_prime = max(1, round(self.k * 0.3))
            selected_indices = np.random.choice(self.M, size=min(k_prime, self.M), replace=False)
            chromosome[selected_indices] = 1
            population.append(chromosome)

            # 初始化变异因子
            self.mutation_factors[u] = np.random.uniform(0.1, 1.0)

        best_fitness = -np.inf
        best_solution = None
        no_improvement_count = 0

        for generation in range(self.max_generations):
            # 并行评估适应度
            fitness_scores = self.parallel_evaluate_fitness(population, observation)

            # 更新最佳解
            for i, fitness in enumerate(fitness_scores):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = population[i].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # 早停检查
            if no_improvement_count >= self.convergence_patience:
                logging.debug(f"[POMDP] Early stopping at generation {generation}")
                break

            # 更新LMP参数（方程38-40）
            self._update_lmp_parameters(generation, fitness_scores)

            # 选择和繁殖
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elite_size = max(1, self.population_size // 4)
            new_population = [population[i] for i in sorted_indices[:elite_size]]

            # 生成后代
            offspring_count = 0
            while len(new_population) < self.population_size:
                parent1_idx = sorted_indices[np.random.randint(elite_size)]
                parent2_idx = sorted_indices[np.random.randint(elite_size)]
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # 自适应稀疏性保持交叉（方程32-35）
                child = self.adaptive_sparsity_preserved_crossover(
                    parent1, parent2,
                    fitness_scores[parent1_idx], fitness_scores[parent2_idx],
                    min(fitness_scores), max(fitness_scores)
                )

                # LPM基础稀疏性保持变异（方程36-42）
                child = self.lmp_based_sparsity_preserved_mutation(child, generation, offspring_count)

                new_population.append(child)
                offspring_count += 1

            population = new_population

        return best_solution if best_solution is not None else population[0]

    def adaptive_sparsity_preserved_crossover(self, parent1, parent2, f1, f2, fmin, fmax):
        """
        自适应稀疏性保持交叉算子（方程32-35）
        严格按照论文理论实现
        """
        child = np.zeros_like(parent1, dtype=int)

        # 计算自适应交叉概率（方程35）
        if fmax - fmin > 1e-8:
            pc = self.pmin + (self.pmax - self.pmin) * (1 - abs(f1 - f2) / (fmax - fmin))
        else:
            pc = self.pmax

        # 计算XOR和相关值
        xor_result = np.logical_xor(parent1, parent2).astype(int)
        h = np.sum(xor_result)  # 要交换的总位数

        if h == 0:
            return parent1.copy()

        # 计算a*和b*
        a_star = np.sum(parent2 * xor_result)  # X中0位且Y中1位的数量
        b_star = np.sum(parent1 * xor_result)  # X中1位且Y中0位的数量

        # 计算归一化因子κ（方程34）
        if a_star > 0 and b_star > 0:
            kappa = h / (2 * a_star) + h / (2 * b_star)

            # 计算概率（方程34）
            p01 = (pc * h) / (2 * a_star * kappa)
            p10 = (pc * h) / (2 * b_star * kappa)

            # 确保概率有效
            p01 = min(1.0, max(0.0, p01))
            p10 = min(1.0, max(0.0, p10))
        else:
            p01 = p10 = 0.0

        # 应用交叉操作（方程32）
        for i in range(len(parent1)):
            if parent1[i] == 0 and parent2[i] == 1:
                if np.random.random() < p01:
                    child[i] = 1
                else:
                    child[i] = 0
            elif parent1[i] == 1 and parent2[i] == 0:
                if np.random.random() < p10:
                    child[i] = 0
                else:
                    child[i] = 1
            else:
                child[i] = parent1[i]

        return child

    def lmp_based_sparsity_preserved_mutation(self, chromosome, generation, chromosome_idx):
        """
        LPM基础稀疏性保持变异算子（方程36-42）
        严格按照论文理论实现
        """
        mutated = chromosome.copy().astype(int)
        M = len(chromosome)

        # 根据LPM计算变异概率（方程41）
        if generation <= self.g1:
            # 第一阶段：使用归一化变异因子
            pm = self._normalize_mutation_factor(self.mutation_factors.get(chromosome_idx, 1.0))
        elif generation <= self.g2:
            # 第二阶段：线性
            pm = self.alpha1 * generation + self.beta4
        else:
            # 第三阶段：常数
            pm = self.beta5

        # 确保pm在有效范围内
        pm = min(1.0, max(0.0, pm))

        # 计算0位和1位数量
        a = np.sum(chromosome == 0)  # 0位数量
        b = np.sum(chromosome == 1)  # 1位数量

        if a == 0 or b == 0:
            return mutated

        # 计算归一化因子ν（方程37）
        nu = M / (2 * a) + M / (2 * b)

        # 计算概率（方程37）
        p0 = (pm * M) / (2 * a * nu)  # 0翻转为1的概率
        p1 = (pm * M) / (2 * b * nu)  # 1翻转为0的概率

        # 确保概率有效
        p0 = min(1.0, max(0.0, p0))
        p1 = min(1.0, max(0.0, p1))

        # 应用变异操作（方程36）
        for i in range(M):
            if chromosome[i] == 0:
                if np.random.random() < p0:
                    mutated[i] = 1
            else:  # chromosome[i] == 1
                if np.random.random() < p1:
                    mutated[i] = 0

        # 确保稀疏性约束（如果超过k，移除多余的1）
        if np.sum(mutated) > self.k:
            ones_indices = np.where(mutated == 1)[0]
            excess = len(ones_indices) - self.k
            if excess > 0:
                remove_indices = np.random.choice(ones_indices, size=excess, replace=False)
                mutated[remove_indices] = 0

        return mutated

    def _update_lmp_parameters(self, generation, fitness_scores):
        """更新LMP参数（方程38-40）"""
        if len(fitness_scores) == 0:
            return

        U = len(fitness_scores)
        sum_fitness = sum(fitness_scores)
        max_fitness = max(fitness_scores)

        if max_fitness <= -1e6:  # 避免无效适应度
            return

        # 更新tau_upper_g（方程40）
        if max_fitness > 0:
            tau_upper_g = self.beta3 * sum_fitness / max_fitness * (
                    self.tau_upper_max - generation * self.tau_upper_unit
            )
        else:
            tau_upper_g = self.tau_upper_max - generation * self.tau_upper_unit

        tau_upper_g = max(0.1, tau_upper_g)

        # 更新tau_g（方程39）
        if tau_upper_g > 0:
            dtau_dg = self.beta2 * self.tau_g * (1 - sum(self.mutation_factors.values()) / tau_upper_g)
            self.tau_g = max(0.1, self.tau_g + dtau_dg)

        # 更新每个染色体的变异因子（方程38）
        for u in range(min(U, len(self.mutation_factors))):
            if u in self.mutation_factors and self.tau_g > 0:
                dru_dg = self.beta1 * self.mutation_factors[u] * (
                        np.log(max(self.tau_g, 1e-8)) - np.log(max(self.mutation_factors[u], 1e-8))
                )
                self.mutation_factors[u] = max(0.01, min(10.0, self.mutation_factors[u] + dru_dg))

    def _normalize_mutation_factor(self, r_u_g):
        """归一化变异因子到[0,1]范围"""
        return min(1.0, max(0.0, r_u_g / 10.0))

    def compute_state_probability(self, state):
        """使用傅里叶近似计算状态概率"""
        prob = 0
        for i in range(self.Theta):
            prob += self.fourier_coeffs_c[i] * np.sin(2 * np.pi * i * state)
            prob += self.fourier_coeffs_d[i] * np.cos(2 * np.pi * i * state)
        return max(0, prob)

    def solve_pomdp(self, observation):
        """求解POMDP并返回最优动作"""
        # 使用观察更新信念状态
        self.adaptive_particle_filtering(observation, 0)

        # 找到最优状态子集
        optimal_subset = self.evolutionary_state_subset_optimization(observation)

        # 计算两种动作的期望奖励
        reward_monitor = self.compute_expected_reward(optimal_subset, observation, action=0)
        reward_check = self.compute_expected_reward(optimal_subset, observation, action=1)

        logging.info(f"[POMDP] Rewards - Monitor: {reward_monitor:.6f}, Check: {reward_check:.6f}")

        return 1 if reward_check > reward_monitor else 0

    def compute_expected_reward(self, state_subset, observation, action):
        """计算给定动作的期望奖励"""
        selected_states = np.where(state_subset == 1)[0] / self.M
        total_reward = 0

        for state in selected_states:
            state_prob = self.compute_state_probability(state)

            if action == 0:  # monitor
                reward = -self.Ca * state
            else:  # check
                reward = -self.CL - self.CR * state

            total_reward += state_prob * reward

        return total_reward

    def _get_fitness_params(self):
        """获取适应度计算所需的参数"""
        return {
            'k': self.k,
            'M': self.M,
            'Ca': self.Ca,
            'CL': self.CL,
            'CR': self.CR,
            'Theta': self.Theta,
            'fourier_coeffs_c': self.fourier_coeffs_c.copy(),
            'fourier_coeffs_d': self.fourier_coeffs_d.copy()
        }

    @staticmethod
    def _evaluate_fitness_worker(args):
        """静态方法用于并行计算适应度"""
        chromosome, observation, params = args

        if np.sum(chromosome) == 0 or np.sum(chromosome) > params['k']:
            return -np.inf

        selected_states = np.where(chromosome == 1)[0] / params['M']
        total_reward = 0

        for state in selected_states:
            # 使用傅里叶系数计算状态概率
            state_prob = 0
            for i in range(params['Theta']):
                state_prob += params['fourier_coeffs_c'][i] * np.sin(2 * np.pi * i * state)
                state_prob += params['fourier_coeffs_d'][i] * np.cos(2 * np.pi * i * state)
            state_prob = max(0, state_prob)

            # 计算两种动作的奖励
            reward_monitor = -params['Ca'] * state
            reward_check = -params['CL'] - params['CR'] * state

            max_reward = max(reward_monitor, reward_check)
            total_reward += state_prob * max_reward

        # 大小惩罚
        size_penalty = -0.01 * np.sum(chromosome)
        return total_reward + size_penalty