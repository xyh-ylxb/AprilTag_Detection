import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import cv2
import numpy as np
import logging
from infrared_edge_detector.camera import InfraredCamera

try:
    import apriltag
except ImportError:
    raise ImportError("请先安装 apriltag 库: pip install apriltag")

def is_camera_in_use(device='/dev/video0'):
    # 返回True表示有进程占用
    return os.system(f"fuser {device} > /dev/null 2>&1") == 0

def monitor_camera_usage(device='/dev/video0', interval=0.5):
    """
    实时输出相机是否被占用，每隔interval秒检测一次，Ctrl+C退出
    """
    import time
    try:
        print(f"开始监控 {device} 是否被占用，按 Ctrl+C 退出...")
        while True:
            in_use = is_camera_in_use(device)
            status = "占用中" if in_use else "空闲"
            print(f"{time.strftime('%H:%M:%S')} - {device}: {status}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n监控已停止。")

def main(device_id=0, width=640, height=480, fps=10, capture_count=5):
    """
    :param capture_count: 每次采集的图片数量
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("apriltag_scanner")

    detector = apriltag.Detector()
    logger.info(f"每隔1秒采集{capture_count}张图片，检测到Apriltag即停止，未检测到则本次采集失败")
    logger.info("按 'Q' 退出，按 'S' 保存当前图片")
    
    frame_count = 0
    last_capture_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # 每隔1秒采集一次
            if current_time - last_capture_time >= 1.0:
                frame_count += 1
                last_capture_time = current_time
                
                # logger.info(f"=== 第 {frame_count} 次采集 ===")
                detected_tags = []
                display_img = None
                camera = InfraredCamera(device_id=device_id, width=width, height=height, fps=fps)
                if not camera.open():
                    logger.error(f"无法打开相机设备 {device_id}")
                    continue
                try:
                    for i in range(capture_count):
                        frame = camera.read_frame()
                        if frame is None:
                            # logger.warning(f"未能读取到相机帧（第{i+1}张），跳过...")
                            continue

                        # 转为8位灰度图
                        if frame.dtype == np.uint16:
                            img8 = (frame / 256).astype(np.uint8)
                        elif frame.dtype == np.uint8:
                            img8 = frame
                        else:
                            img8 = frame.astype(np.uint8)

                        if len(img8.shape) == 3:
                            gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
                            display_img = img8.copy()
                        else:
                            gray = img8
                            display_img = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

                        # 检测Apriltag
                        results = detector.detect(gray)
                        if results:
                            detected_tags = results
                            # logger.info(f"检测到 {len(results)} 个Apriltag（第{i+1}张）")
                            break
                        # else:
                            # logger.info(f"第{i+1}张未检测到Apriltag")
                finally:
                    camera.close()

                # 处理检测结果
                if detected_tags:
                    # logger.info(f"最终检测到 {len(detected_tags)} 个Apriltag:")
                    for tag in detected_tags:
                        # 绘制标签边框
                        corners = tag.corners.astype(np.int32)
                        cv2.polylines(display_img, [corners], True, (0, 255, 0), 2)
                        
                        # 计算朝向角度
                        pt0, pt1 = tag.corners[0], tag.corners[1]
                        dx, dy = pt1[0] - pt0[0], pt1[1] - pt0[1]
                        theta = np.arctan2(dy, dx) * 180 / np.pi
                        theta = (theta + 360) % 360
                        
                        # 显示ID和角度
                        text = f"ID:{tag.tag_id} {theta:.1f}°"
                        cv2.putText(display_img, text, 
                                   (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 在终端输出详细信息
                        logger.info(f"  ID: {tag.tag_id}, 角度: {theta:.1f}")
                else:
                    logger.info(f"连续{capture_count}张图片均未检测到Apriltag")

                # 添加时间戳、帧计数和检测信息
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                status_text = f"Frame: {frame_count} | Time: {timestamp}"
                if detected_tags:
                    status_text += f" | Tags: {len(detected_tags)}"
                else:
                    status_text += f" | No Tag"
                if display_img is not None:
                    cv2.putText(display_img, status_text, 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # 显示图像
                    cv2.imshow("Apriltag Scanner (interval open/close)", display_img)
                    # 按键处理
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("用户按Q退出")
                        break
                    elif key == ord('s'):
                        # 保存当前帧
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"apriltag_capture_{timestamp}_frame{frame_count}.jpg"
                        cv2.imwrite(filename, display_img)
                        logger.info(f"已保存图片: {filename}")
                else:
                    # 没有可显示图片
                    pass
            # else:
            #     # 检查相机是否被占用，并输出状态
            #     in_use = is_camera_in_use(f'/dev/video{device_id}')
            #     status = "占用中" if in_use else "空闲"
            #     print(f"{time.strftime('%H:%M:%S')} - /dev/video{device_id}: {status}")
            # 短暂休眠，减少CPU占用
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("用户中断，退出...")
    finally:
        cv2.destroyAllWindows()
        logger.info(f"程序已结束，共采集 {frame_count} 帧")

if __name__ == "__main__":
    # main()
    # 如需监控相机占用情况，取消下行注释
    # monitor_camera_usage('/dev/video0')
    main() 