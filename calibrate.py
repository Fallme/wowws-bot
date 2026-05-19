"""屏幕坐标校准工具 - 首次运行时使用"""

import sys
import time
import yaml
import numpy as np
import mss
import cv2


def capture_screen():
    """截取全屏"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # 主显示器
        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def select_region(img, region_name):
    """让用户用鼠标框选区域"""
    print(f"\n请用鼠标框选 [{region_name}] 区域，然后按 Enter 确认，按 ESC 取消")
    print("操作: 按住左键拖动选择区域")

    roi = cv2.selectROI(f"选择 {region_name}", img, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        return None

    x, y, w, h = roi
    print(f"  已选择: x={x}, y={y}, w={w}, h={h}")
    return {"x": x, "y": y, "w": w, "h": h}


def main():
    print("=" * 50)
    print("  战舰世界Bot - 屏幕坐标校准工具")
    print("=" * 50)
    print("\n请确保战舰世界已打开并显示港口界面")
    print("按 Enter 截取屏幕...")
    input()

    img = capture_screen()
    h, w = img.shape[:2]
    print(f"屏幕分辨率: {w}x{h}")

    # 显示截图
    cv2.imshow("Screen Preview - Press any key to continue", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 校准各区域
    regions = {}

    print("\n--- 开始校准 ---")

    region_names = [
        ("minimap", "小地图 (右下角)"),
        ("main_gun_reload", "主炮装填条 (屏幕下方中间偏左)"),
        ("health_bar", "血条 (屏幕下方中间)"),
        ("torpedo_reload", "鱼雷装填条 (主炮装填条下方)"),
        ("distance_text", "距离数字 (准心下方)"),
    ]

    for name, desc in region_names:
        region = select_region(img, f"{desc}")
        if region:
            regions[name] = region
        else:
            print(f"  跳过 {name}")

    # 校准按钮位置
    print("\n--- 校准按钮位置 ---")
    print("请点击 '战斗' 按钮的位置")
    print("请在截图上点击 '战斗' 按钮，然后按 Enter")
    cv2.imshow("Click Battle Button", img)
    point = cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # 简化：让用户输入坐标
    battle_x = int(input("战斗按钮 X 坐标: ") or "960")
    battle_y = int(input("战斗按钮 Y 坐标: ") or "1000")

    continue_x = int(input("继续按钮 X 坐标: ") or "960")
    continue_y = int(input("继续按钮 Y 坐标: ") or "800")

    # 保存配置
    config = {
        "resolution": {"width": w, "height": h},
        "regions": regions,
        "buttons": {
            "battle": {"x": battle_x, "y": battle_y},
            "continue": {"x": continue_x, "y": continue_y},
        },
    }

    with open("config/screen.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"\n校准完成! 配置已保存到 config/screen.yaml")
    print(f"分辨率: {w}x{h}")
    print(f"校准了 {len(regions)} 个区域")


if __name__ == "__main__":
    main()
