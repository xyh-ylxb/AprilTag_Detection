from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import argparse
import os


def apriltag_to_pdf(image_path, pdf_path, tag_size_cm=1.2):
    """
    生成A4 PDF，将Apriltag图片以指定厘米尺寸居中放置
    :param image_path: 输入图片路径
    :param pdf_path: 输出PDF路径
    :param tag_size_cm: 标签边长（厘米）
    """
    # 页面尺寸
    page_width, page_height = A4
    tag_size_pt = tag_size_cm * cm  # 1cm = 28.346pt

    # 计算居中位置
    x = (page_width - tag_size_pt) / 2
    y = (page_height - tag_size_pt) / 2

    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)  # 白底

    # 加载图片
    img = ImageReader(image_path)
    c.drawImage(img, x, y, width=tag_size_pt, height=tag_size_pt, preserveAspectRatio=True, mask='auto')

    c.showPage()
    c.save()
    print(f"已生成PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="将Apriltag图片以指定尺寸放入A4 PDF")
    parser.add_argument('--image', type=str, required=True, help='Apriltag图片路径')
    parser.add_argument('--output', type=str, default='apriltag_a4.pdf', help='输出PDF路径')
    parser.add_argument('--size', type=float, default=1.2, help='标签边长（厘米）')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"图片文件不存在: {args.image}")
        return

    apriltag_to_pdf(args.image, args.output, args.size)

if __name__ == "__main__":
    main() 