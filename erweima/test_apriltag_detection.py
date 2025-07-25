import cv2
import numpy as np
import argparse
import os

try:
    import apriltag
except ImportError:
    raise ImportError("请先安装 apriltag 库: pip install apriltag")

def detect_apriltag_in_image(image_path, headless=False):
    """
    检测图片中的Apriltag
    :param image_path: 图片路径
    :param headless: 是否无头模式（不显示图片）
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    print(f"图片尺寸: {img.shape}")
    print(f"图片数据类型: {img.dtype}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"灰度图尺寸: {gray.shape}")
    print(f"灰度图数值范围: {gray.min()} - {gray.max()}")
    
    # 创建检测器
    detector = apriltag.Detector()
    
    # 检测Apriltag
    results = detector.detect(gray)
    
    if results:
        print(f"检测到 {len(results)} 个Apriltag:")
        for i, tag in enumerate(results):
            print(f"  标签 {i+1}:")
            print(f"    ID: {tag.tag_id}")
            print(f"    角点坐标: {tag.corners.tolist()}")
            
            # 计算朝向角度
            pt0, pt1 = tag.corners[0], tag.corners[1]
            dx, dy = pt1[0] - pt0[0], pt1[1] - pt0[1]
            theta = np.arctan2(dy, dx) * 180 / np.pi
            print(f"    朝向角度: {theta:.1f}°")
            
            if not headless:
                # 在图片上绘制检测结果
                cv2.polylines(img, [tag.corners.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(img, f"ID:{tag.tag_id}", 
                           (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        print("未检测到Apriltag")
        # 保存调试图片
        debug_path = "debug_gray.png"
        cv2.imwrite(debug_path, gray)
        print(f"已保存调试图片: {debug_path}")
    
    if not headless:
        # 显示结果
        cv2.imshow("Apriltag Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="测试Apriltag检测功能")
    parser.add_argument('image_path', help='Apriltag图片路径')
    parser.add_argument('--headless', action='store_true', help='无头模式（不显示图片）')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"图片文件不存在: {args.image_path}")
        return
    
    detect_apriltag_in_image(args.image_path, headless=args.headless)

if __name__ == "__main__":
    main() 