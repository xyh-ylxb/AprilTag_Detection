import os
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from erweima.pose_estimator import PoseEstimator, TagDetection
from infrared_edge_detector.camera import InfraredCamera

try:
    import apriltag
except ImportError:
    raise ImportError("请先安装 apriltag 库: pip install apriltag")

def main(device_id=0, width=640, height=480, fps=10, capture_count=5):
    estimator = PoseEstimator()
    detector = apriltag.Detector()
    print("按 'Q' 退出，按 'S' 保存当前图片")

    frame_count = 0
    last_capture_time = time.time()

    try:
        while True:
            current_time = time.time()
            if current_time - last_capture_time >= 1.0:
                frame_count += 1
                last_capture_time = current_time
                detected_tags = []
                display_img = None
                camera = InfraredCamera(device_id=device_id, width=width, height=height, fps=fps)
                if not camera.open():
                    print(f"无法打开相机设备 {device_id}")
                    continue
                try:
                    for i in range(capture_count):
                        frame = camera.read_frame()
                        if frame is None:
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
                            break
                finally:
                    camera.close()
                # 转换为TagDetection对象
                detections = []
                print(f"本帧检测到 {len(detected_tags)} 个tag")
                for tag in detected_tags:
                    print(f"  tag_id: {tag.tag_id}")
                    angle = (np.arctan2(tag.corners[1][1] - tag.corners[0][1], tag.corners[1][0] - tag.corners[0][0]) * 180 / np.pi + 360) % 360
                    detection = TagDetection(
                        tag_id=tag.tag_id,
                        corners=tag.corners,
                        center=np.mean(tag.corners, axis=0),
                        angle=angle
                    )
                    detections.append(detection)
                # 输出相机在圆盘坐标系下的朝向（融合）
                if detections:
                    img_cx, img_cy = width / 2, height / 2
                    thetas = []
                    weights = []
                    for detection in detections:
                        theta_disk = estimator.get_camera_orientation_on_disk(detection)
                        cx, cy = detection.center
                        dist = np.hypot(cx - img_cx, cy - img_cy)
                        weight = 1.0 / (dist + 1e-6)  # 距离越近权重越大
                        thetas.append(theta_disk)
                        weights.append(weight)
                    # 加权向量平均融合角度
                    thetas_rad = np.radians(thetas)
                    avg_sin = np.average(np.sin(thetas_rad), weights=weights)
                    avg_cos = np.average(np.cos(thetas_rad), weights=weights)
                    theta_fused = (np.degrees(np.arctan2(avg_sin, avg_cos)) + 360) % 360
                    print(f"融合后的相机在圆盘坐标系下的朝向: {theta_fused:.1f}°")
                # 位姿估计
                pose = estimator.estimate_pose_multi_tags(detections)
                # 打印位姿信息
                if pose:
                    print(f"位姿估计: X={pose.x:.3f}m, Y={pose.y:.3f}m, θ={pose.theta:.1f}°, 置信度={pose.confidence:.2f}")
                else:
                    print("未能估计出位姿")
                # 可视化
                if pose:
                    vis_img = estimator.visualize_pose(display_img, pose, detections)
                else:
                    vis_img = display_img
                # 添加状态信息
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                status_text = f"Frame: {frame_count} | Time: {timestamp}"
                if detections:
                    status_text += f" | Tags: {len(detections)}"
                else:
                    status_text += f" | No Tag"
                if pose:
                    status_text += f" | Pose: ({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.1f}°)"
                if vis_img is not None:
                    cv2.putText(vis_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Scan & Estimate", vis_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户按Q退出")
                        break
                    elif key == ord('s'):
                        save_name = f"scan_estimate_{time.strftime('%Y%m%d_%H%M%S')}_frame{frame_count}.jpg"
                        cv2.imwrite(save_name, vis_img)
                        print(f"已保存图片: {save_name}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("用户中断，退出...")
    finally:
        cv2.destroyAllWindows()
        print(f"程序已结束，共采集 {frame_count} 帧")

if __name__ == "__main__":
    main() 