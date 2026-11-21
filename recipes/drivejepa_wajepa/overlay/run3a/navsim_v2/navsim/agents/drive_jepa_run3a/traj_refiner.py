import torch.nn as nn

from .agent_look import AgentLook
from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MLP
from .drive_jepa_config import DriveJEPAConfig


class Traj_refiner(nn.Module):
    def __init__(self, config: DriveJEPAConfig):
        super().__init__()

        self.poses_num = config.num_poses
        self.state_size = 3

        self.traj_bev = config.traj_bev
        self.b2d = config.b2d

        if self.traj_bev:
            self.Bev_refiner = Bev_refiner(config, config.proposal_num, self.poses_num, config.traj_proposal_query)

        self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, self.state_size)

        # run 3A: agent look (paddock-ops/run3a/CLAUDE.md), one block shared by the 4 rounds; the global look is removed
        self.agent_look = AgentLook(config.tf_d_model, 8, config.agent_look_k, config.agent_look_cut)

    def forward(self, bev_feature, image_feature, agents=None):
        proposals = self.traj_decoder(bev_feature).reshape(bev_feature.shape[0], -1, self.poses_num, self.state_size)

        if self.traj_bev:
            bev_feature = self.Bev_refiner(proposals, bev_feature, image_feature)

        if agents is not None:
            bev_feature = self.agent_look(bev_feature, proposals, agents)

        return bev_feature, proposals
