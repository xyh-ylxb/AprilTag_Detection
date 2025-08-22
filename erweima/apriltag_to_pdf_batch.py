from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import os
import argparse


def batch_apriltag_to_pdf(
    image_dir="/home/yunhao.xia/code/AprilTag_Detection/large_apriltags",
    pdf_path="batch_apriltags_a4.pdf",
    tag_size_cm=1.2,
    start_num=1,
    end_num=30
):
    """
    一次性将Apriltag二维码图片以指定尺寸排列到A4 PDF上，并在下方显示编号。
    :param image_dir: 图片所在文件夹
    :param pdf_path: 输出PDF路径
    :param tag_size_cm: 标签边长（厘米）
    :param start_num: 起始编号
    :param end_num: 结束编号（包含）
    """
    page_width, page_height = A4
    tag_size_pt = tag_size_cm * cm
    margin_x = 1.5 * cm  # 左右边距
    margin_y = 2 * cm    # 上下边距
    spacing_x = 1.0 * cm # 水平间距
    spacing_y = 1.2 * cm # 垂直间距（含编号空间）

    # 计算每行、每列可放置的二维码数量
    max_cols = int((page_width - 2 * margin_x + spacing_x) // (tag_size_pt + spacing_x))
    max_rows = int((page_height - 2 * margin_y + spacing_y) // (tag_size_pt + spacing_y))
    per_page = max_cols * max_rows

    c = canvas.Canvas(pdf_path, pagesize=A4)

    total_tags = end_num - start_num + 1
    for idx in range(total_tags):
        tag_num = start_num + idx
        imagenumber = str(tag_num).zfill(5)
        image_path = os.path.join(image_dir, f"apriltag_{imagenumber}.png")
        col = idx % max_cols
        row = (idx // max_cols) % max_rows
        page = idx // per_page
        if idx > 0 and idx % per_page == 0:
            c.showPage()
        # 重新计算row/col在当前页的位置
        col = (idx % per_page) % max_cols
        row = (idx % per_page) // max_cols
        x = margin_x + col * (tag_size_pt + spacing_x)
        y = page_height - margin_y - (row + 1) * (tag_size_pt + spacing_y) + spacing_y
        # 绘制二维码
        if os.path.exists(image_path):
            img = ImageReader(image_path)
            c.drawImage(img, x, y, width=tag_size_pt, height=tag_size_pt, preserveAspectRatio=True, mask='auto')
        else:
            c.setFillColorRGB(1, 0, 0)
            c.rect(x, y, tag_size_pt, tag_size_pt, fill=1)
            c.setFillColorRGB(0, 0, 0)
        # 绘制编号
        c.setFont("Helvetica-Bold", 10)
        text_width = c.stringWidth(imagenumber, "Helvetica-Bold", 10)
        text_x = x + (tag_size_pt - text_width) / 2
        text_y = y - 0.4 * cm
        c.drawString(text_x, text_y, imagenumber)
    c.showPage()
    c.save()
    print(f"已生成批量PDF: {pdf_path}，共{total_tags}个二维码")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将Apriltag图片以指定尺寸放入A4 PDF，并在下方显示编号")
    parser.add_argument('--image_dir', type=str, default='/home/yunhao.xia/code/AprilTag_Detection/large_apriltags', help='Apriltag图片文件夹')
    parser.add_argument('--output', type=str, default='batch_apriltags_a4.pdf', help='输出PDF路径')
    parser.add_argument('--size', type=float, default=1.2, help='标签边长（厘米）')
    parser.add_argument('--start', type=int, default=1, help='起始编号')
    parser.add_argument('--end', type=int, default=30, help='结束编号（包含）')
    args = parser.parse_args()

    batch_apriltag_to_pdf(
        image_dir=args.image_dir,
        pdf_path=args.output,
        tag_size_cm=args.size,
        start_num=args.start,
        end_num=args.end
    ) 