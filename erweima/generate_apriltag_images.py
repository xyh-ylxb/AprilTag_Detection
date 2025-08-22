import os
import argparse
try:
    import apriltag
except ImportError:
    raise ImportError("请先安装 apriltag 库: pip install apriltag")
import cv2
import numpy as np

def generate_apriltag(tag_family: str, tag_id: int, size: int = 300) -> np.ndarray:
    """
    生成单个Apriltag图像
    :param tag_family: 标签系列，如 'tag36h11'
    :param tag_id: 标签ID
    :param size: 图片尺寸（像素）
    :return: 标签图像（uint8）
    """
    # apriltag-python 只提供检测，不提供生成，所以用opencv自带的生成器
    try:
        detector = cv2.apriltag_AprilTagDetector_create()
        generator = cv2.apriltag_AprilTagGenerator_create(tag_family)
        img = generator.generate(tag_id, size)
        return img
    except Exception:
        # 若opencv不支持，提示用户用官方工具生成
        raise RuntimeError("当前环境不支持Apriltag生成。建议用 https://github.com/AprilRobotics/apriltag-imgs 或在线工具生成。");

def main():
    parser = argparse.ArgumentParser(description="批量生成Apriltag图片")
    parser.add_argument('--family', type=str, default='tag36h11', help='Apriltag系列（如tag36h11）')
    parser.add_argument('--start', type=int, default=0, help='起始ID')
    parser.add_argument('--end', type=int, default=42, help='结束ID（包含）')
    parser.add_argument('--size', type=int, default=300, help='图片尺寸（像素）')
    parser.add_argument('--outdir', type=str, default='apriltag_imgs', help='输出文件夹')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for tag_id in range(args.start, args.end + 1):
        try:
            img = generate_apriltag(args.family, tag_id, args.size)
            out_path = os.path.join(args.outdir, f"{args.family}_{tag_id}.png")
            cv2.imwrite(out_path, img)
            print(f"已生成: {out_path}")
        except Exception as e:
            print(f"生成ID={tag_id}失败: {e}")

if __name__ == "__main__":
    main() 