import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import PDMSimulator
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.evaluate.pdm_score import fast_transform_trajectory
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
    ego_is_two_frame_extended_comfort,
)
from .bevformer.simple_image_encoder import ImgEncoder
from .bevformer.resnet_image_encoder import ResNetImgEncoder
from .drive_jepa_config import DriveJEPAConfig
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner

class WAJEPATokenProjector(nn.Module):
    """In place of the ViT (wa-jepa/CLAUDE.md 3.2): cached frozen WA-JEPA tokens (B, 4, 512, 1024) -> Linear(1024, 256)
    (as ImgEncoder.projector) and a learned camera embedding (as cams_embeds in bevformer/image_encoder.py)."""

    def __init__(self, config: DriveJEPAConfig, num_cams: int = 4):
        super().__init__()
        self.projector = nn.Linear(1024, config.tf_d_model)
        self.cams_embeds = nn.Parameter(torch.randn([num_cams, config.tf_d_model]))


proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)

class DriveJEPAModel(nn.Module):
    def __init__(self, config: DriveJEPAConfig):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3

        if config.use_wajepa_tokens:
            self._backbone = WAJEPATokenProjector(config)
        else:
            self._backbone = ResNetImgEncoder(config) if config.use_resnet else ImgEncoder(config)
        self.use_global_look = config.use_wajepa_tokens  # False only for pre-flight check 5 (kv=None)
        if os.environ.get("WAJEPA_GLOBAL_LOOK") == "0":  # ablation (19 Oct): global look off at eval, kv=None
            self.use_global_look = False
        # Drive-JEPA's lidar2img is for its 0.4-scale 768x432 image; the tokens come from rows 28:-28 of the frame at
        # 512x256: u' = u / 1.5, v' = (v - 11.2) / 1.6 (the shift multiplies depth). Applied once in forward.
        self.register_buffer("to_512", torch.tensor(
            [[2 / 3, 0, 0, 0], [0, 0.625, -7, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32), persistent=False)
        self.hist_encoding = nn.Linear(11, config.tf_d_model)
        self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)

        shared_refiner = Traj_refiner(config)
        self._trajectory_head = nn.ModuleList([shared_refiner for _ in range(config.ref_num)])

        self.scorer = Scorer(config)

        self.transform = self.make_transform()

    def make_transform(self):
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([normalize])

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:, -1]

        batch_size = ego_status.shape[0]
        kv = None
        if self._config.use_wajepa_tokens:
            # all 4 cameras (b0 f0 l0 r0): no f0-only slice; converted once into the 512x256 model image, then used by
            # the rays (all 4) and the footprint look (f0 l0 r0); the cache keeps Drive-JEPA's matrix
            lidar2img = self.to_512 @ features["lidar2img"].float()
            x = features["wajepa_tokens"]  # float16 from the cache; autocast runs the Linear in fp16 anyway
            if not torch.is_autocast_enabled():
                x = x.to(self._backbone.projector.weight.dtype)
            tokens = self._backbone.projector(x)
            if self.use_global_look:
                kv = self._trajectory_head[0].global_look.kv(tokens, lidar2img)  # before the camera embedding
            feat = tokens + self._backbone.cams_embeds[None, :, None, :].to(tokens.dtype)  # (B, 4, 512, C)
            fp = list(self._config.footprint_cams)  # footprint look: f0 l0 r0
            feat_flatten = feat[:, fp].permute(1, 2, 0, 3)  # (cams, 512, B, C)
            spatial_shape = torch.as_tensor([[16, 32]], dtype=torch.long, device=feat.device)
            level_start_index = spatial_shape.new_zeros((1,))
            img_metas = {"lidar2img": lidar2img[:, fp],
                         "img_shape": lidar2img.new_tensor([256.0, 512.0, 3.0]).expand(batch_size, len(fp), 3)}
            image_feature = (feat_flatten, spatial_shape, level_start_index, {"img_metas": img_metas})
        else:
            if self._config.use_resnet:
                # 4 surround cameras (b0, f0, l0, r0), already normalized in the feature builder
                camera_feature = features["camera_feature"]
            else:
                # front camera f0 only, 2 frames
                features["lidar2img"] = features["lidar2img"][:, 1:2]
                cam_f_2 = features["camera_feature_2"]
                cam_f_1 = features["camera_feature_1"]
                cam_f_2 = self.transform(cam_f_2)
                cam_f_1 = self.transform(cam_f_1)
                camera_feature = torch.cat([cam_f_2[:, None], cam_f_1[:, None]], dim=1)
            image_feature = self._backbone(camera_feature, img_metas=features)  # b,64,64,64
        ego_feature = self.hist_encoding(ego_status)[:, None]
        bev_feature = ego_feature + self.init_feature.weight[None]

        proposal_list = []
        for _, refine in enumerate(self._trajectory_head):
            bev_feature, proposals = refine(bev_feature, image_feature, kv)
            proposal_list.append(proposals)

        proposals = proposal_list[-1]

        pred_logit, pred_agents_states, pred_area_logit, bev_semantic_map, agent_states, agent_labels = self.scorer(
            proposals, bev_feature
        )

        output = {}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list
        output["pred_logit"] = pred_logit
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
        
        if 'past_ego_simulated_states' in features and features['past_ego_simulated_states'] is not None:
            pdm_score = self.calibrate_score(features, proposals, pdm_score)

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output
    
    def calibrate_score(self, features, proposals, pdm_score):
        past_ego_simulated_states = features['past_ego_simulated_states']
        observation_interval = features['observation_interval']
        metric_cache: MetricCache = features['metric_cache']
        initial_ego_state = metric_cache.ego_state

        if past_ego_simulated_states is None:
            return pdm_score
        
        # TODO: 1. simulate states; 2. calculate EC; 3. Calibrate pdm_score
        trajectory_states = []
        for model_trajectory in proposals.squeeze():
            pred_states = fast_transform_trajectory(Trajectory(model_trajectory.float().cpu().numpy()), simulator.proposal_sampling, initial_ego_state)
            trajectory_states.append(pred_states)

        trajectory_states = np.stack(trajectory_states, axis=0)
        extended = np.zeros((*trajectory_states.shape[:-1], 11), dtype=trajectory_states.dtype)
        extended[:, :, :3] = trajectory_states
        
        simulated_states = simulator.simulate_proposals(extended, initial_ego_state)#32,41,11
        
        for idx, sim_state in enumerate(simulated_states):
            interval_length = 0.1
            overlap_start = int(observation_interval / interval_length)

            current_states = past_ego_simulated_states 
            next_states = sim_state 

            # Ensure they have the same shape
            assert (
                current_states.shape == next_states.shape
            ), "Trajectories must be of equal length"

            # Extract only the overlapping part
            current_states_overlap = current_states[overlap_start:]
            next_states_overlap = next_states[:-overlap_start]

            # Define corresponding time points for overlap
            n_overlap = current_states_overlap.shape[
                0
            ]  # Compute the actual number of overlapping steps
            time_point_s = (
                np.arange(n_overlap) * interval_length
            )  # Generate aligned time steps

            # Compute two-frame extended comfort
            two_frame_comfort = ego_is_two_frame_extended_comfort(
                current_states_overlap[None, :],
                next_states_overlap[None, :],
                time_point_s,
            )[0].astype(np.float64)
            pdm_score[:, idx] = (14.0 * pdm_score[:, idx] + 2.0 * two_frame_comfort) / 16.0

        return pdm_score
