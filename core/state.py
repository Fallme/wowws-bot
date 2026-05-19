"""游戏状态数据类"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Enemy:
    """敌方目标"""
    x: float          # 小地图x坐标 (像素)
    y: float          # 小地图y坐标 (像素)
    distance: float = 0.0   # 距离 (米)
    heading: float = 0.0    # 航向角度
    speed: float = 0.0      # 航速
    destroyed: bool = False # 是否被击沉


@dataclass
class GameState:
    """游戏状态"""
    # 基本状态
    in_battle: bool = False        # 是否在战斗中
    battle_ended: bool = False     # 战斗是否结束
    victory: Optional[bool] = None # 胜利/失败

    # 玩家状态
    health: float = 1.0            # 血量百分比 (0~1)
    main_gun_ready: bool = False   # 主炮是否装填好
    torpedo_ready: bool = False    # 鱼雷是否装填好
    smoke_available: bool = False  # 烟雾是否可用

    # 目标状态
    enemies: list = field(default_factory=list)      # 敌方目标列表
    locked_target: Optional[Enemy] = None             # 当前锁定的目标
    secondary_locked: bool = False                    # 副炮是否已锁定

    # 鱼雷检测
    torpedo_incoming: bool = False  # 是否有敌方鱼雷来袭

    # 距离
    target_distance: float = 0.0   # 当前目标距离 (米)

    # 小地图数据
    minimap_enemies: list = field(default_factory=list)  # 小地图上的敌人坐标
