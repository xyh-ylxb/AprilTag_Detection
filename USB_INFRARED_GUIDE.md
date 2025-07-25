# USB红外相机接入与使用指南

## 🎯 USB红外相机实际接入流程

### 第一步：系统环境准备
由于检测到缺少 `v4l2-ctl` 工具，先安装必要依赖：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install v4l-utils

# CentOS/RHEL
sudo yum install v4l-utils

# 验证安装
which v4l2-ctl
```

### 第二步：物理连接与识别
```bash
# 插入USB红外相机后立即执行：
lsusb | grep -E "(camera|Camera|IR|Thermal|红外|热成像)"

# 查看新出现的视频设备
ls -la /dev/video* 2>/dev/null || echo "暂无视频设备"

# 查看系统日志
dmesg | tail -20
```

### 第三步：权限与驱动检查
```bash
# 检查当前用户是否在video组
groups | grep video

# 如果没有，添加到video组
sudo usermod -a -G video $USER
sudo chmod 666 /dev/video*  # 临时权限修复

# 重新登录或执行
newgrp video
```

### 第四步：相机功能验证

#### 基础检测（无需v4l2-ctl）
```bash
python3 -c "
from infrared_edge_detector import InfraredCamera
for i in range(10):
    try:
        cam = InfraredCamera(i)
        if cam.open():
            info = cam.get_camera_info()
            print(f'设备{i}: {info}')
            cam.close()
    except: pass
"
```

#### 使用我们的检测工具
```bash
# 快速检测
python3 quick_usb_check.py

# 交互式检测（推荐）
python3 interactive_usb_detector.py

# 基础功能测试
python3 test_basic.py
```

### 第五步：实际红外边缘检测

#### 方案A：一键启动
```bash
# 直接使用默认设备
python main.py

# 指定设备ID（根据检测结果调整）
python main.py --device 0
```

#### 方案B：配置文件调整
编辑 `config/infrared_config.yaml` 根据实际相机参数调整：
```yaml
camera:
  device_id: 0    # 根据实际检测结果调整
  width: 640      # 根据相机支持的分辨率调整
  height: 480
  fps: 10         # 根据相机能力调整
```

#### 方案C：无红外相机时的模拟测试
```bash
python demo_headless.py  # 使用模拟红外数据
```

## 🔧 常见问题解决

### 问题1: 设备无法打开
```bash
# 检查设备权限
ls -la /dev/video0

# 临时修复权限
sudo chmod 666 /dev/video0

# 永久修复
sudo usermod -a -G video $USER
```

### 问题2: 无v4l2-ctl工具
```bash
sudo apt install v4l-utils
```

### 问题3: OpenCV无法识别设备
```bash
# 检查OpenCV版本
python -c "import cv2; print(cv2.__version__)"

# 测试基础相机访问
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### 问题4: 红外相机温度校准
```bash
# 运行温度校准工具
python3 -c "
from infrared_edge_detector import InfraredProcessor
import numpy as np

# 模拟红外数据处理
processor = InfraredProcessor()
mock_data = np.random.randint(20000, 40000, (480, 640), dtype=np.uint16)
processed = processor.process_frame(mock_data)
print('温度范围:', processed['temperature_range'])
"
```

## 📊 红外相机接入验证步骤

### 步骤1：硬件连接检查
```bash
# 物理连接后立即执行
echo "=== 1. USB设备列表 ==="
lsusb | grep -E "(Camera|camera|IR|Thermal|红外|热成像)" || echo "未发现红外相机"

echo "=== 2. 视频设备 ==="
ls -la /dev/video* 2>/dev/null || echo "无视频设备"

echo "=== 3. 系统识别 ==="
dmesg | grep -E "(uvcvideo|video|camera)" | tail -5
```

### 步骤2：软件兼容性测试
```bash
echo "=== 4. OpenCV兼容性 ==="
python3 -c "
import cv2
import numpy as np
print('OpenCV版本:', cv2.__version__)

# 测试所有可能的设备
for i in range(5):
    try:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f'设备{i}: 分辨率{frame.shape}, 类型{frame.dtype}')
            cap.release()
    except:
        pass
"
```

### 步骤3：红外特化测试
```bash
echo "=== 5. 红外边缘检测测试 ==="
python test_basic.py
```

## 🚀 一键启动脚本

创建一键启动脚本 `run_infrared.sh`：

```bash
#!/bin/bash
echo "红外相机边缘检测启动器"

# 检查依赖
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检测设备
echo "正在检测设备..."
python3 -c "
from infrared_edge_detector import InfraredCamera
import cv2

devices = []
for i in range(10):
    try:
        cam = InfraredCamera(i)
        if cam.open():
            info = cam.get_camera_info()
            devices.append((i, info))
            cam.close()
    except:
        pass

if devices:
    print('找到设备:')
    for dev_id, info in devices:
        print(f'  设备{dev_id}: {info.get(\"width\", 0)}x{info.get(\"height\", 0)}')
    print('启动边缘检测...')
    import subprocess
    subprocess.run(['python3', 'main.py', '--device', str(devices[0][0])])
else:
    print('未找到相机设备')
    print('运行模拟模式...')
    subprocess.run(['python3', 'demo_headless.py'])
"
```

## 📋 实际使用场景指南

### 场景1：工业红外检测
```bash
# 连接红外热像仪后立即开始检测
python main.py --device 0 --width 640 --height 480 --fps 15
```

### 场景2：安防监控部署
```bash
# 后台运行模式
nohup python main.py --device 0 > infrared.log 2>&1 &
```

### 场景3：科研数据采集
```bash
# 高分辨率模式
python main.py --device 0 --width 1280 --height 720 --fps 10
```

## 🎯 成功接入的判断标准

当您的USB红外相机成功接入时，应该能看到：
1. ✅ 设备被系统识别 (出现 `/dev/videoX`)
2. ✅ OpenCV能够打开设备并读取图像
3. ✅ 我们的检测脚本显示"✅ 连接成功"
4. ✅ main.py能够正常运行并开始边缘检测

现在您已经拥有完整的USB红外相机接入和处理系统，随时准备处理实际的USB红外相机数据！