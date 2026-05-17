from __future__ import print_function

import torch
import torch.nn.functional as F
import logging
import random
from copy import deepcopy

from utils import utils
from clients import *


class FreeRiderClient(Client):
    def __init__(self, cid, model, dataLoader, optimizer,
                 criterion=F.cross_entropy, device='cpu', inner_epochs=1,
                 mean=0.0, std=0.01, attack_prob=0.9):
        super(FreeRiderClient, self).__init__(cid, model, dataLoader,
                                              optimizer, criterion,
                                              device, inner_epochs)
        self.mean = mean
        self.std = std
        self.attack_prob = attack_prob  # Attack probability for this client
        self.is_attacking = False  # Initially not attacking
        logging.info(f"Initialized FreeRiderClient {cid} with mean={mean}, std={std}, attack_prob={attack_prob}")

    def decide_attack(self):
        """Decide whether to initiate an attack based on attack probability"""
        if not self.is_attacking:
            if random.random() < self.attack_prob:
                self.is_attacking = True
                logging.info(f"FreeRiderClient {self.cid} initiated attack")

        return self.is_attacking

    def update_param(self, update_type='common'):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'

        # Check if client should attack in this round
        self.decide_attack()

        if not self.is_attacking:
            super().update_param(update_type)
        else:
            newState = self.model.state_dict()
            self.raw_stateChange = {}  # Initialize raw stateChange storage

            if update_type == "common":
                trainable_parameter = utils.getTrainableParameters(self.model)
                for p in self.originalState:
                    diff = newState[p] - self.originalState[p]
                    self.stateChange[p] = diff
                    self.raw_stateChange[p] = diff.clone()
                    if p not in trainable_parameter:
                        continue
                    std = torch.ones(self.stateChange[p].shape) * self.std
                    noise = torch.normal(mean=self.mean, std=std)
                    self.stateChange[p] = noise

            elif update_type == "mudhog":
                trainable_parameter = utils.getTrainableParameters(self.model)
                for p in self.originalState:
                    diff = newState[p] - self.originalState[p]
                    self.stateChange[p] = diff
                    self.raw_stateChange[p] = diff.clone()
                    if p not in trainable_parameter:
                        continue
                    std = torch.ones(self.stateChange[p].shape) * self.std
                    noise = torch.normal(mean=self.mean, std=std)
                    self.stateChange[p] = noise
                    self.sum_hog[p] += self.stateChange[p]

                    K_ = len(self.hog_avg)
                    if K_ == 0:
                        self.avg_delta[p] = self.stateChange[p]
                    elif K_ < self.K_avg:
                        self.avg_delta[p] = (self.avg_delta[p] * K_ + self.stateChange[p]) / (K_ + 1)
                    else:
                        self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p]) / self.K_avg
                self.hog_avg.append(deepcopy(self.stateChange))

            elif update_type == "pomdpfl":
                trainable_parameter = utils.getTrainableParameters(self.model)
                for p in self.originalState:
                    diff = newState[p] - self.originalState[p]
                    self.stateChange[p] = diff
                    self.raw_stateChange[p] = diff.clone()
                    if p not in trainable_parameter:
                        continue
                    std = torch.ones(self.stateChange[p].shape) * self.std
                    noise = torch.normal(mean=self.mean, std=std)
                    self.stateChange[p] = noise

                # 更新分类层EMA（使用受攻击的参数）
                self._update_classifier_ema()

            self.isTrained = False

    def resubmit(self):
        """当被检测为自由搭便车攻击时重新提交正常参数"""
        self.is_attacking = False  # Stop the attack
        logging.info(f"FreeRiderClient {self.cid} was detected and resubmitted valid model")
        return self.raw_stateChange


class PoisoningClient(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.cross_entropy,
                 device='cpu', inner_epochs=1, source_label=9, target_label=7, attack_prob=0.9):
        super(PoisoningClient, self).__init__(cid, model, dataLoader, optimizer,
                                              criterion, device, inner_epochs)
        self.source_label = source_label
        self.target_label = target_label
        self.attack_prob = attack_prob  # Attack probability for this client
        self.is_attacking = False  # Initially not attacking
        self.detected = False  # Initially not detected
        logging.info(
            f"Initialized Poisoning Client {cid} with label flipping {source_label} -> {target_label}, attack_prob={attack_prob}")

    def decide_attack(self):
        """Decide whether to initiate an attack based on attack probability"""
        if not self.is_attacking and not self.detected:
            if random.random() < self.attack_prob:
                self.is_attacking = True
                logging.info(f"PoisoningClient {self.cid} initiated attack")

        return self.is_attacking

    def data_transform(self, data, target):
        if self.detected:
            return data, target

        self.decide_attack()

        if self.is_attacking:
            target_ = torch.tensor(list(map(lambda x: self.target_label if x == self.source_label else x, target)))
            return data, target_
        else:
            return data, target

    def train(self):
        if self.detected:
            logging.info(f"PoisoningClient {self.cid} is permanently banned, skipping training")
            self.isTrained = True
            return
        super().train()

    def set_detected(self):
        self.is_attacking = False  # Stop the attack
        self.detected = True  # Mark as detected
        logging.info(f"PoisoningClient {self.cid} was detected and permanently blocked from training")