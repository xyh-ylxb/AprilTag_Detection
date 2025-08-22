import cv2
import numpy as np
import os
import argparse

def resize_apriltag_image(input_path, output_path, target_size=800):
    """
    将小尺寸的Apriltag图片放大到指定尺寸
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_size: 目标尺寸（像素）
    """
    # 读取原图
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图片: {input_path}")
        return False
    
    # 放大图片，使用INTER_NEAREST保持边缘清晰
    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # 保存放大后的图片
    cv2.imwrite(output_path, resized)
    print(f"已生成: {output_path} ({target_size}x{target_size})")
    return True

def batch_resize_apriltags(input_dir, output_dir, target_size=300, start_id=0, end_id=10):
    """
    批量处理Apriltag图片
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有png文件
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    png_files.sort()
    
    # 处理指定ID范围的文件
    for i, filename in enumerate(png_files):
        if start_id <= i <= end_id:
            input_path = os.path.join(input_dir, filename)
            output_filename = f"apriltag_{i:05d}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            resize_apriltag_image(input_path, output_path, target_size)

def main():
    parser = argparse.ArgumentParser(description="生成大尺寸Apriltag图片")
    parser.add_argument('--input_dir', type=str, default='apriltag-imgs/tag36h11', 
                        help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, default='large_apriltags', 
                        help='输出文件夹路径')
    parser.add_argument('--size', type=int, default=800, 
                        help='目标尺寸（像素）')
    parser.add_argument('--start', type=int, default=0, 
                        help='起始ID')
    parser.add_argument('--end', type=int, default=43, 
                        help='结束ID')
    
    args = parser.parse_args()
    
    batch_resize_apriltags(args.input_dir, args.output_dir, args.size, args.start, args.end)
    print(f"\n批量处理完成！输出目录: {args.output_dir}")

if __name__ == "__main__":
    main() 