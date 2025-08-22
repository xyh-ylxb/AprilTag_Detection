from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import argparse
import os
import re


def apriltag_to_pdf(image_path, pdf_path, tag_size_cm=1.2):
    """
    生成A4 PDF，将Apriltag图片以指定厘米尺寸居中放置，并在下方显示imagename
    :param image_path: 输入图片路径
    :param pdf_path: 输出PDF路径
    :param tag_size_cm: 标签边长（厘米）
    """
    # 页面尺寸
    page_width, page_height = A4
    tag_size_pt = tag_size_cm * cm  # 1cm = 28.346pt

    # 提取imagename（从文件名中提取数字编号）
    filename = os.path.basename(image_path)
    # 匹配文件名中的数字编号，如00001, 001, 1等
    match = re.search(r'(\d+)', filename)
    if match:
        imagenumber = match.group(1).zfill(5)  # 补齐5位，如00001
    else:
        imagenumber = "00000"  # 默认值

    # 计算二维码位置（居中偏上，为下方文字留空间）
    x = (page_width - tag_size_pt) / 2
    y = (page_height - tag_size_pt) / 2 + 2 * cm  # 向上偏移2cm

    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)  # 白底

    # 加载并绘制二维码图片
    img = ImageReader(image_path)
    c.drawImage(img, x, y, width=tag_size_pt, height=tag_size_pt, preserveAspectRatio=True, mask='auto')

    # 在下方添加imagename
    c.setFillColorRGB(0, 0, 0)  # 黑色文字
    c.setFont("Helvetica-Bold", 16)  # 设置字体和大小
    
    # 计算文字位置（居中，在二维码下方）
    text_width = c.stringWidth(imagenumber, "Helvetica-Bold", 16)
    text_x = (page_width - text_width) / 2
    text_y = y - 1 * cm  # 在二维码下方1cm处
    
    c.drawString(text_x, text_y, imagenumber)

    c.showPage()
    c.save()
    print(f"已生成PDF: {pdf_path}，imagename: {imagenumber}")


def main():
    parser = argparse.ArgumentParser(description="将Apriltag图片以指定尺寸放入A4 PDF，并在下方显示imagename")
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