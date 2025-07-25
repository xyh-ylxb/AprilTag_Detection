import time
import cv2
import numpy as np
import logging
from infrared_edge_detector.camera import InfraredCamera
from erweima.qrcode_util import decode_qrcode


def main(device_id=0, width=640, height=480, fps=10):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("usb_qrcode_scanner")

    camera = InfraredCamera(device_id=device_id, width=width, height=height, fps=fps)
    if not camera.open():
        logger.error(f"无法打开相机设备 {device_id}")
        return

    # logger.info(f"已打开相机: {device_id}")
    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                logger.warning("未能读取到相机帧，重试中...")
                time.sleep(1)
                continue

            # 转为8位灰度图（二维码检测更稳定）
            if frame.dtype == np.uint16:
                img8 = (frame / 256).astype(np.uint8)
            elif frame.dtype == np.uint8:
                img8 = frame
            else:
                img8 = frame.astype(np.uint8)

            # OpenCV要求3通道或单通道
            if len(img8.shape) == 2:
                img_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img8

            # 保存临时图片
            tmp_path = "_tmp_qr_scan.png"
            cv2.imwrite(tmp_path, img_bgr)

            try:
                info, angle = decode_qrcode(tmp_path)
                logger.info(f"二维码内容: {info}")
                logger.info(f"二维码相对角度: {angle}")
            except Exception as e:
                logger.info(f"未检测到二维码: {e}")

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("用户中断，退出...")
    finally:
        camera.close()
        # logger.info("相机已关闭")


if __name__ == "__main__":
    main() 