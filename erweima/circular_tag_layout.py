#!/usr/bin/env python3
"""
1.25m半径圆板二维码布局生成器
生成99个二维码的精确坐标和可视化
"""

import math
import json
import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 二维码布局参数
CIRCLE_RADIUS = 1.25  # 圆板半径(m)
TAG_SIZE = 0.012      # 二维码实际尺寸(m)
GRID_SPACING = 0.20   # 六边形网格间距(m)
TOTAL_TAGS = 99       # 总二维码数量

class CircularTagLayout:
    """圆板二维码布局管理器"""
    
    def __init__(self):
        self.tags = []  # 存储所有二维码信息
        
    def generate_layout(self) -> List[Dict]:
        """生成99个二维码的精确布局，螺旋式空间连续编号"""
        tags = []
        # 先生成所有点，带圈号和角度
        all_points = []
        # 中心点
        all_points.append({
            "circle": 0,
            "angle": 0.0,
            "x": 0.0,
            "y": 0.0,
            "radius": 0.0
        })
        # 定义统一的角度间隔 - 使用30度间隔确保对齐
        angle_step = 30  # 30度间隔
        base_angles = [i * angle_step for i in range(12)]  # 0, 30, 60, ..., 330
        # 8圈布局 - 使用统一角度确保对齐
        ring_configs = [
            {"radius": 0.20, "angles": [0, 60, 120, 180, 240, 300]},      # 0.2m半径，6个点
            {"radius": 0.35, "angles": base_angles},                       # 0.35m半径，12个点
            {"radius": 0.50, "angles": base_angles + [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]},  # 0.5m半径，24个点
            {"radius": 0.65, "angles": base_angles + [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]},  # 0.65m半径，24个点
            {"radius": 0.80, "angles": base_angles + [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]},  # 0.8m半径，24个点
            {"radius": 0.95, "angles": base_angles + [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]},  # 0.95m半径，24个点
            {"radius": 1.10, "angles": base_angles + [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]},  # 1.1m半径，24个点
            {"radius": 1.20, "angles": base_angles}                        # 1.2m半径，12个点
        ]
        for circle_idx, config in enumerate(ring_configs, start=1):
            radius = config["radius"]
            angles = config["angles"]
            for angle_deg in angles:
                angle_rad = math.radians(angle_deg)
                x = radius * math.cos(angle_rad)
                y = radius * math.sin(angle_rad)
                all_points.append({
                    "circle": circle_idx,
                    "angle": angle_deg,
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "radius": round(radius, 2)
                })
        # 螺旋式排序：先按circle递增，再按angle递增
        all_points_sorted = sorted(all_points, key=lambda p: (p["circle"], p["angle"]))
        # 重新分配id，从1开始
        for i, pt in enumerate(all_points_sorted):
            tags.append({
                "id": i + 1,  # 从1开始编号
                "x": pt["x"],
                "y": pt["y"],
                "radius": pt["radius"],
                "angle": 0.0  # 所有二维码上边与x轴平行
            })
        self.tags = tags
        return tags
    
    def save_layout(self, filename: str = "tag_layout.json"):
        """保存布局到JSON文件"""
        layout_data = {
            "metadata": {
                "circle_radius": CIRCLE_RADIUS,
                "tag_size": TAG_SIZE,
                "total_tags": len(self.tags),
                "grid_spacing": GRID_SPACING
            },
            "tags": self.tags
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        print(f"布局已保存到: {filename}")
    
    def visualize_layout(self, save_path: str = "circular_layout.png"):
        """生成圆板可视化布局图"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # 绘制圆板边界
        circle = Circle((0, 0), CIRCLE_RADIUS,
                       fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # 绘制二维码位置
        x_coords = [tag["x"] for tag in self.tags]
        y_coords = [tag["y"] for tag in self.tags]
        
        # 绘制每个二维码
        for tag in self.tags:
            # 二维码位置点
            ax.plot(tag["x"], tag["y"], 'ro', markersize=8, zorder=3)
            
            # 添加ID标签
            ax.annotate(str(tag["id"]), 
                       (tag["x"], tag["y"]), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8, 
                       color='blue',
                       fontweight='bold')
            # 绘制朝向箭头（所有二维码上边与y轴平行，朝向0度）
            arrow_length = 0.04  # 箭头长度
            angle_rad = math.radians(tag["angle"] + 90)  # 0度为y轴正方向
            dx = arrow_length * math.cos(angle_rad)
            dy = arrow_length * math.sin(angle_rad)
            ax.arrow(tag["x"], tag["y"], dx, dy, head_width=0.02, head_length=0.02, fc='g', ec='g', zorder=4)
        
        # 绘制网格线
        ring_radii = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.2]
        for i, radius in enumerate(ring_radii):
            circle_grid = Circle((0, 0), radius,
                               fill=False,
                               color='gray',
                               linewidth=0.5,
                               linestyle='--',
                               alpha=0.5,
                               label=f"r={radius:.2f}m" if i == 0 else None)  # 只为第一个添加label，后续用手动图例
            ax.add_patch(circle_grid)
        # 手动添加图例
        legend_labels = [f"r={r:.2f}m" for r in ring_radii]
        legend_handles = [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1, alpha=0.7) for _ in ring_radii]
        ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=10, title='Ring Radius', title_fontsize=11)
        
        # 添加笛卡尔坐标网格线（每0.1m一条）
        grid_step = 0.1
        grid_range = np.arange(-1.3, 1.31, grid_step)
        for gx in grid_range:
            ax.plot([gx, gx], [-1.3, 1.3], color='lightgray', linewidth=0.5, linestyle='-', zorder=0)
        for gy in grid_range:
            ax.plot([-1.3, 1.3], [gy, gy], color='lightgray', linewidth=0.5, linestyle='-', zorder=0)
        
        # 设置图形属性
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('AprilTag Layout', fontsize=16)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # 添加比例尺
        ax.plot([-1.2, -1.0], [-1.2, -1.2], 'k-', linewidth=3)
        ax.text(-1.1, -1.15, '0.2m', ha='center', fontsize=10)
        
        # 添加角度定义说明（英文）
        ax.text(0, -1.28, "Orientation: 0° is +Y axis, increases clockwise, max 360°", fontsize=12, color='green', ha='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"布局可视化已保存到: {save_path}")
    
    def generate_coordinate_map(self) -> Dict[int, Tuple[float, float]]:
        """生成ID到坐标的映射"""
        return {tag["id"]: (tag["x"], tag["y"]) for tag in self.tags}
    
    def check_coverage(self) -> bool:
        """验证布局覆盖率 - 基于相机参数"""
        # 相机参数
        camera_height = 0.3  # 相机高度0.3m
        fov_degrees = 50     # 视野角度50度
        fov_radians = math.radians(fov_degrees)
        view_radius = camera_height * math.tan(fov_radians / 2)  # 视野半径
        
        print(f"相机参数分析:")
        print(f"  相机高度: {camera_height}m")
        print(f"  视野角度: {fov_degrees}度")
        print(f"  视野半径: {view_radius:.3f}m")
        
        # 创建测试点网格
        test_points = []
        step = 0.02  # 2cm步长，更精细的测试
        for x in np.arange(-1.25, 1.25, step):
            for y in np.arange(-1.25, 1.25, step):
                if x**2 + y**2 <= 1.25**2:  # 在圆内
                    test_points.append((x, y))
        
        # 检查每个测试点到最近二维码的距离
        uncovered = []
        coverage_stats = {
            "total_points": len(test_points),
            "covered_points": 0,
            "uncovered_points": 0,
            "min_distance": float('inf'),
            "max_distance": 0,
            "avg_distance": 0
        }
        
        total_distance = 0
        
        for px, py in test_points:
            min_dist = min(
                math.sqrt((px - tag["x"])**2 + (py - tag["y"])**2)
                for tag in self.tags
            )
            total_distance += min_dist
            
            if min_dist <= view_radius:
                coverage_stats["covered_points"] += 1
            else:
                uncovered.append((px, py, min_dist))
                coverage_stats["uncovered_points"] += 1
            
            coverage_stats["min_distance"] = min(coverage_stats["min_distance"], min_dist)
            coverage_stats["max_distance"] = max(coverage_stats["max_distance"], min_dist)
        
        coverage_stats["avg_distance"] = total_distance / len(test_points)
        
        coverage_rate = coverage_stats["covered_points"] / coverage_stats["total_points"] * 100
        
        print(f"\n覆盖率分析:")
        print(f"  总测试点数: {coverage_stats['total_points']}")
        print(f"  覆盖点数: {coverage_stats['covered_points']}")
        print(f"  未覆盖点数: {coverage_stats['uncovered_points']}")
        print(f"  覆盖率: {coverage_rate:.1f}%")
        print(f"  最小距离: {coverage_stats['min_distance']:.3f}m")
        print(f"  最大距离: {coverage_stats['max_distance']:.3f}m")
        print(f"  平均距离: {coverage_stats['avg_distance']:.3f}m")
        
        if uncovered:
            print(f"\n未覆盖区域分析:")
            max_uncovered = max(uncovered, key=lambda x: x[2])
            print(f"  最大未覆盖距离: {max_uncovered[2]:.3f}m")
            print(f"  最大未覆盖位置: ({max_uncovered[0]:.3f}, {max_uncovered[1]:.3f})")
            
            # 统计未覆盖区域的分布
            edge_uncovered = sum(1 for x, y, _ in uncovered if x**2 + y**2 > 1.0**2)
            center_uncovered = len(uncovered) - edge_uncovered
            print(f"  边缘区域未覆盖: {edge_uncovered}个点")
            print(f"  中心区域未覆盖: {center_uncovered}个点")
        
        # 检查是否满足全覆盖要求
        if coverage_rate >= 99.5:
            print(f"\n✅ 布局满足全覆盖要求 (覆盖率 >= 99.5%)")
            return True
        else:
            print(f"\n❌ 布局不满足全覆盖要求 (覆盖率 < 99.5%)")
            return False

def main():
    """主函数：生成完整布局"""
    layout = CircularTagLayout()
    
    # 生成布局
    tags = layout.generate_layout()
    print(f"生成二维码布局: {len(tags)}个")
    
    # 保存布局
    layout.save_layout("erweima/circular_tag_layout.json")
    
    # 验证覆盖率
    layout.check_coverage()
    
    # 生成可视化
    layout.visualize_layout("erweima/circular_layout.png")
    
    # 打印前10个二维码坐标
    print("\n前10个二维码坐标:")
    for i, tag in enumerate(tags[:10]):
        print(f"ID {tag['id']}: ({tag['x']:.3f}, {tag['y']:.3f})")
    
    return layout

if __name__ == "__main__":
    layout = main()