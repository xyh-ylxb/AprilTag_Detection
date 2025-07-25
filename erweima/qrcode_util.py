import qrcode
import cv2
import numpy as np
import json
from typing import Tuple, Dict
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _has_pyzbar = True
except ImportError:
    _has_pyzbar = False


def generate_qrcode(data: dict, filename: str = 'qrcode.png'):
    """
    生成包含序号、数字和角度信息的二维码
    :param data: 例如 {'id': 123, 'value': 456, 'angle': 0}
    :param filename: 保存的二维码图片名
    """
    content = json.dumps(data, ensure_ascii=False)
    img = qrcode.make(content)
    img.save(filename)
    print(f"二维码已保存到: {filename}")


def decode_qrcode(image_path: str) -> Tuple[Dict, int]:
    """
    解码二维码内容，并判断二维码相对角度（0/90/180/270）
    :param image_path: 二维码图片路径
    :return: (二维码内容dict, 角度int)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {image_path}")

    # 使用OpenCV检测二维码及其角度
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if not data:
        # 兼容pyzbar
        if _has_pyzbar:
            decoded = pyzbar_decode(img)
            if decoded:
                data = decoded[0].data.decode('utf-8')
                points = np.array([d.polygon for d in decoded][0], dtype=np.float32)
            else:
                raise ValueError("未检测到二维码")
        else:
            raise ValueError("未检测到二维码，且未安装pyzbar")

    # 解析内容
    try:
        info = json.loads(data)
    except Exception:
        info = {'raw': data}

    # 角度判断
    angle = 0
    # 修正：兼容OpenCV返回的三维points数组
    if points is not None:
        pts = np.array(points)
        if pts.shape == (1, 4, 2):
            pts = pts[0]
        if pts.shape == (4, 2):
            # OpenCV返回顺序: 左上、右上、右下、左下
            v1 = pts[1] - pts[0]  # 左上->右上
            dx, dy = v1[0], v1[1]
            theta = np.arctan2(dy, dx) * 180 / np.pi
            if -45 <= theta < 45:
                angle = 0
            elif 45 <= theta < 135:
                angle = 90
            elif theta >= 135 or theta < -135:
                angle = 180
            else:
                angle = 270
        else:
            angle = None
    else:
        angle = None

    return info, angle


if __name__ == "__main__":
    # 示例用法
    # 生成二维码
    # data = {'id': 115, 'pos': [1,5]}
    # generate_qrcode(data, '115_15.png')

    # # 解码二维码
    info, angle = decode_qrcode('113_13.png')
    print(f"二维码内容: {info}")
    print(f"二维码相对角度: {angle}") 