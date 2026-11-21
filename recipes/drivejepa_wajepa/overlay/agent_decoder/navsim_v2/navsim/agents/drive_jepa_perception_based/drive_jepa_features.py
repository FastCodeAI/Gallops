from enum import IntEnum
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import numpy.typing as npt

import torch
from torchvision import transforms

from shapely import affinity
from shapely.geometry import Polygon, LineString

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from navsim.agents.drive_jepa_perception_based.drive_jepa_config import DriveJEPAConfig
from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from .bevformer.bev_feature_build import _get_bev_feature, lidar2img_from_calibration



class DriveJEPAFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: DriveJEPAConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drive_jepa_resnet_feature" if self._config.use_resnet else "pad_feature"

    def _get_camera_feature(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras
        
        # Crop to ensure 2:1 aspect ratio
        f0 = cameras[-1].cam_f0.image[28:-28]

        # stitch l0, f0, r0 images
        resized_image_1 = cv2.resize(f0, (512, 256))
        
        # Crop to ensure 2:1 aspect ratio
        f0 = cameras[-2].cam_f0.image[28:-28]

        # stitch l0, f0, r0 images
        resized_image_2 = cv2.resize(f0, (512, 256))
        tensor_image_1 = transforms.ToTensor()(resized_image_1)
        tensor_image_2 = transforms.ToTensor()(resized_image_2)
        
        features = {
            'camera_feature_1': tensor_image_1,
            'camera_feature_2': tensor_image_2,
        }
        return features 
    
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = _get_bev_feature(agent_input)
        if not self._config.use_resnet:
            features.update(self._get_camera_feature(agent_input))
        #features["lidar_feature"] = self._get_lidar_feature(agent_input)

        ego_feature_list=[]

        for ego_status in agent_input.ego_statuses:
            if ego_status is None:
                continue
            pose=torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)

            ego_feature=torch.cat([pose,velocity, acceleration, driving_command], dim=-1)
            ego_feature_list.append(ego_feature)

        features["ego_status"] =torch.stack(ego_feature_list)

        return features

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)


class DriveJEPA4VFeatureBuilder(AbstractFeatureBuilder):
    """Light features for the 4-camera WA-JEPA planner (wa-jepa/CLAUDE.md 1.1): v2's feature builder without
    'camera_feature' and without _get_camera_feature. lidar2img (4, 4, 4; b0 f0 l0 r0) and img_shape (4, 3) are exactly
    what _get_bev_feature returns (its 0.4-scale matrix; img_shape is the padded (448, 768, 3)), computed from the camera
    calibration only, so no image is loaded or rescaled; ego_status as DriveJEPAFeatureBuilder."""

    IMG_SHAPE = (448.0, 768.0, 3.0)

    def __init__(self, config: DriveJEPAConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "drivejepa4v_feature"

    @staticmethod
    def light_features(cams: Dict[str, Any], ego_statuses) -> Dict[str, torch.Tensor]:
        """cams: {"cam_b0": (sensor2lidar_rotation, sensor2lidar_translation, intrinsics), ...} at time t."""
        features = {
            "lidar2img": lidar2img_from_calibration([cams[c] for c in ("cam_b0", "cam_f0", "cam_l0", "cam_r0")]),
            "img_shape": torch.tensor([DriveJEPA4VFeatureBuilder.IMG_SHAPE] * 4, dtype=torch.float32),
        }
        ego = []
        for ego_status in ego_statuses:
            if ego_status is None:
                continue
            ego.append(torch.cat([torch.tensor(ego_status.ego_pose, dtype=torch.float32),
                                  torch.tensor(ego_status.ego_velocity, dtype=torch.float32),
                                  torch.tensor(ego_status.ego_acceleration, dtype=torch.float32),
                                  torch.tensor(ego_status.driving_command, dtype=torch.float32)], dim=-1))
        features["ego_status"] = torch.stack(ego)
        return features

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        cameras = agent_input.cameras[-1]
        cams = {c: (getattr(cameras, c).sensor2lidar_rotation, getattr(cameras, c).sensor2lidar_translation,
                    getattr(cameras, c).intrinsics) for c in ("cam_b0", "cam_f0", "cam_l0", "cam_r0")}
        return self.light_features(cams, agent_input.ego_statuses)


class WAJEPATokenBuilder(AbstractFeatureBuilder):
    """WA-JEPA tokens (wa-jepa/CLAUDE.md 1.2): {"wajepa_tokens": float16 (4, 512, 1024)}. Training reads the cached file;
    navtest eval runs it live (the frozen encoder is built on first use)."""

    _ENCODERS = {}  # one frozen encoder per process and device: navsim's compute_trajectory builds new builders per scene

    def __init__(self, config: DriveJEPAConfig, device: str = "cuda"):
        self._config = config
        self._device = device

    def get_unique_name(self) -> str:
        return "wajepa_tokens"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        from .bevformer.wajepa import build_wajepa_encoder, encode_wajepa

        if self._device not in WAJEPATokenBuilder._ENCODERS:
            WAJEPATokenBuilder._ENCODERS[self._device], _ = build_wajepa_encoder(self._device)
        return {"wajepa_tokens": encode_wajepa(agent_input, WAJEPATokenBuilder._ENCODERS[self._device])}


class DriveJEPATargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: DriveJEPAConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "pad_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._config.trajectory_sampling.num_poses
            ).poses
        )
        # frame_idx = scene.scene_metadata.num_history_frames - 1
        # annotations = scene.frames[frame_idx].annotations
        # ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        # agent_states, agent_labels = self._compute_agent_targets(annotations)
        # bev_semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)

        return {
            "trajectory": trajectory,
            # "agent_states": agent_states,
            # "agent_labels": agent_labels,
            # "bev_semantic_map": bev_semantic_map,
            "token":scene.scene_metadata.initial_token
        }

    def _compute_agent_targets(self, annotations: Annotations) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2D agent bounding boxes in ego coordinates
        :param annotations: annotation dataclass
        :return: tuple of bounding box values and labels (binary)
        """

        max_agents = self._config.num_bounding_boxes
        agent_states_list: List[npt.NDArray[np.float32]] = []

        def _xy_in_lidar(x: float, y: float, config: DriveJEPAConfig) -> bool:
            return (config.lidar_min_x <= x <= config.lidar_max_x) and (
                config.lidar_min_y <= y <= config.lidar_max_y
            )

        for box, name in zip(annotations.boxes, annotations.names):
            box_x, box_y, box_heading, box_length, box_width = (
                box[0],
                box[1],
                box[6],
                box[3],
                box[4],
            )

            if name == "vehicle" and _xy_in_lidar(box_x, box_y, self._config):
                agent_states_list.append(
                    np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                )

        agents_states_arr = np.array(agent_states_list)

        # filter num_instances nearest
        agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)

        if len(agents_states_arr) > 0:
            distances = np.linalg.norm(agents_states_arr[..., slice(0, 1 + 1)], axis=-1)
            argsort = np.argsort(distances)[:max_agents]

            # filter detections
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True

        return torch.tensor(agent_states), torch.tensor(agent_labels)

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(map_linestring_mask, [points], isClosed=False, color=255, thickness=2)
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(
        self, annotations: Annotations, layers: TrackedObjectType
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)#128,256
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0

    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self, layers=layers
        )
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


AGENT_CLASSES = ("vehicle", "pedestrian", "bicycle")  # class index 0, 1, 2; the agent decoder adds 3 = none


class DriveJEPAAgentTargetBuilder(AbstractTargetBuilder):
    """Agent-decoder targets (agent decoder spec, 22 Oct): DriveJEPATargetBuilder._compute_agent_targets extended to all
    vehicles, pedestrians and cyclists ('bicycle') at time t with |x| <= 32 m and |y| <= 32 m in the ego frame (behind
    included), nearest first, padded to 64 with a mask. Per agent: class (index into AGENT_CLASSES), box (x, y, z, l, w,
    h, yaw) and velocity (vx, vy), both in the ego frame as the logs store them (velocity absolute). agent_count is the
    number in range before the cut to 64. Run 2's own target builder is unchanged."""

    MAX_AGENTS, RANGE_M = 64, 32.0

    def __init__(self, config: DriveJEPAConfig = None):
        self._config = config

    def get_unique_name(self) -> str:
        return "agent_targets"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        frame_idx = scene.scene_metadata.num_history_frames - 1
        return self.agent_targets(scene.frames[frame_idx].annotations)

    @classmethod
    def agent_targets(cls, annotations: Annotations) -> Dict[str, torch.Tensor]:
        names = np.asarray(annotations.names)
        boxes = np.asarray(annotations.boxes, dtype=np.float32).reshape(-1, BoundingBoxIndex.size())
        vel = np.asarray(annotations.velocity_3d, dtype=np.float32).reshape(-1, 3)
        keep = np.isin(names, AGENT_CLASSES) & (np.abs(boxes[:, BoundingBoxIndex.X]) <= cls.RANGE_M) & (
            np.abs(boxes[:, BoundingBoxIndex.Y]) <= cls.RANGE_M)
        names, boxes, vel = names[keep], boxes[keep], vel[keep]
        order = np.argsort(np.hypot(boxes[:, BoundingBoxIndex.X], boxes[:, BoundingBoxIndex.Y]), kind="stable")
        order = order[: cls.MAX_AGENTS]
        n = len(order)

        agent_class = np.zeros(cls.MAX_AGENTS, dtype=np.int64)
        agent_box = np.zeros((cls.MAX_AGENTS, 7), dtype=np.float32)
        agent_vel = np.zeros((cls.MAX_AGENTS, 2), dtype=np.float32)
        agent_mask = np.zeros(cls.MAX_AGENTS, dtype=bool)
        agent_class[:n] = [AGENT_CLASSES.index(c) for c in names[order]]
        agent_box[:n] = boxes[order, :7]
        agent_vel[:n] = vel[order, :2]
        agent_mask[:n] = True
        return {
            "agent_class": torch.tensor(agent_class),
            "agent_box": torch.tensor(agent_box),
            "agent_vel": torch.tensor(agent_vel),
            "agent_mask": torch.tensor(agent_mask),
            "agent_count": torch.tensor(int(keep.sum())),
        }


class BoundingBox2DIndex(IntEnum):

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
