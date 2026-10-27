import torch.nn as nn

from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MLP
from .drive_jepa_config import DriveJEPAConfig
from .global_look import GlobalLook


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

        # global look (wa-jepa/CLAUDE.md 3.5): one block, shared by the 4 rounds with this refiner
        self.global_look = GlobalLook(config.tf_d_model, 8) if config.use_wajepa_tokens else None

    def forward(self, bev_feature, image_feature, kv=None):
        proposals = self.traj_decoder(bev_feature).reshape(bev_feature.shape[0], -1, self.poses_num, self.state_size)

        if self.traj_bev:
            bev_feature = self.Bev_refiner(proposals, bev_feature, image_feature)

        if kv is not None:
            bev_feature = self.global_look(bev_feature, proposals, kv)

        return bev_feature, proposals
