"""Contains all callable environments in Flow."""

from flow.envs.base_env import Env
from flow.envs.bay_bridge.base import BayBridgeEnv
from flow.envs.bottleneck_env import BottleNeckAccelEnv, BottleneckEnv, \
    DesiredVelocityEnv
from flow.envs.green_wave_env import TrafficLightGridEnv, \
    PO_TrafficLightGridEnv, GreenWaveTestEnv
from flow.envs.loop.lane_changing import LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv
from flow.envs.loop.loop_accel import AccelEnv, AccelMLPGlobalEnv, \
    AccelMLPLocalEnv, AccelCNNEnv, AccelCNNIDMEnv, AccelCNNPIEnv, AccelIDMEnv, \
    AccelPIEnv
from flow.envs.loop.loop_merges import TwoLoopsMergePOEnv, \
    TwoLoopsMergeMLPGlobalEnv, TwoLoopsMergeMLPLocalEnv, TwoLoopsMergeCNNEnv, \
    TwoLoopsMergeCNNIDMEnv, TwoLoopsMergeCNNPIEnv, TwoLoopsMergeIDMEnv, \
    TwoLoopsMergePIEnv
from flow.envs.loop.wave_attenuation import WaveAttenuationEnv, \
    WaveAttenuationPOEnv, WaveAttenuationMLPGlobalEnv, \
    WaveAttenuationMLPLocalEnv, WaveAttenuationCNNEnv, WaveAttenuationCNNIDMEnv, \
    WaveAttenuationCNNPIEnv, WaveAttenuationIDMEnv, WaveAttenuationPIEnv
from flow.envs.merge import WaveAttenuationMergePOEnv
from flow.envs.test import TestEnv

__all__ = [
    "Env", "AccelEnv", "LaneChangeAccelEnv", "LaneChangeAccelPOEnv",
    "GreenWaveTestEnv", "GreenWaveTestEnv", "WaveAttenuationMergePOEnv",
    "TwoLoopsMergePOEnv", "BottleneckEnv", "BottleNeckAccelEnv",
    "WaveAttenuationEnv", "WaveAttenuationPOEnv", "TrafficLightGridEnv",
    "PO_TrafficLightGridEnv", "DesiredVelocityEnv", "TestEnv", "BayBridgeEnv",
    "WaveAttenuationMLPGlobalEnv", "WaveAttenuationMLPLocalEnv",
    "WaveAttenuationCNNEnv", "WaveAttenuationCNNIDMEnv",
    "WaveAttenuationCNNPIEnv", "WaveAttenuationIDMEnv", "WaveAttenuationPIEnv",
    "AccelMLPGlobalEnv", "AccelMLPLocalEnv", "AccelCNNEnv", "AccelCNNIDMEnv",
    "AccelCNNPIEnv", "AccelIDMEnv", "AccelPIEnv",
    "TwoLoopsMergeMLPGlobalEnv", "TwoLoopsMergeMLPLocalEnv",
    "TwoLoopsMergeCNNEnv", "TwoLoopsMergeCNNIDMEnv", "TwoLoopsMergeCNNPIEnv",
    "TwoLoopsMergeIDMEnv", "TwoLoopsMergePIEnv"
]
