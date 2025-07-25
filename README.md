# 红外相机边缘检测系统

基于嵌入式Linux + OpenCV的红外相机边缘检测系统，支持V4L2接口的红外相机。

## 🚀 功能特性

### 核心功能
- **红外数据采集**: 支持16位灰度红外图像采集
- **温度数据转换**: 将16位原始数据转换为可视化温度图像
- **多算法边缘检测**: 
  - Canny边缘检测
  - Sobel梯度检测
  - Laplacian边缘检测
  - 自适应阈值检测
  - 红外特有的温度梯度边缘检测
- **图像预处理**:
  - 时域滤波去噪
  - 高斯模糊降噪
  - CLAHE对比度增强
  - 动态温度范围调整

### 系统特性
- **性能监控**: 实时监控FPS、内存使用、CPU占用、处理延迟
- **配置管理**: YAML配置文件，支持动态参数调整
- **跨平台支持**: 支持ARM和x86_64架构的Linux系统
- **模块化设计**: 清晰的模块分离，易于扩展和维护

## 📁 项目结构

```
infrared_edge_detector/
├── __init__.py           # 包初始化
├── camera.py            # 红外相机采集模块
├── processor.py         # 红外图像预处理模块
├── edge_detector.py     # 边缘检测算法模块
├── monitor.py           # 性能监控模块
└── config.py            # 系统配置管理

config/
└── infrared_config.yaml # 配置文件

main.py                  # 主程序入口
test_basic.py           # 基础功能测试
demo.py                 # 带显示的演示程序
demo_headless.py        # 无头演示程序
```

## 🔧 快速开始

### 环境要求
```bash
pip install opencv-python numpy psutil pyyaml flask
```

### 基础测试
```bash
python test_basic.py
```

### 运行演示
```bash
# 无头演示（不依赖显示环境）
python demo_headless.py

# 带显示的演示（需要图形界面）
python demo.py
```

### 系统运行
```bash
# 相机连接测试
python main.py --test

# 正常运行
python main.py

# 指定配置文件
python main.py -c /path/to/config.yaml
```

## ⚙️ 配置说明

### 相机配置
```yaml
camera:
  device_id: 0          # 相机设备ID
  width: 640           # 图像宽度
  height: 480          # 图像高度
  fps: 10              # 目标帧率
  format: "Y16"        # 红外数据格式
```

### 处理配置
```yaml
processing:
  gaussian_kernel: [5, 5]      # 高斯核大小
  clahe_clip_limit: 2.0        # CLAHE对比度限制
  clahe_tile_size: [8, 8]      # CLAHE网格大小
  temporal_filter_size: 3      # 时域滤波器大小
```

### 边缘检测配置
```yaml
edge_detection:
  algorithm: "canny"           # 边缘检测算法
  canny_low_threshold: 50      # Canny低阈值
  canny_high_threshold: 150    # Canny高阈值
  temperature_threshold: 5.0   # 温度梯度阈值
```

## 📊 性能指标

### 测试结果
- **基础功能测试**: ✅ 全部通过
- **算法兼容性**: ✅ 支持多种边缘检测算法
- **性能表现**: 
  - 模拟数据下达到 7-9 FPS（单线程）
  - 处理延迟 < 20ms
  - 内存使用 < 100MB

## 🎯 使用场景

### 典型应用
1. **工业检测**: 设备热故障检测
2. **安防监控**: 人体检测和跟踪
3. **医疗诊断**: 体表温度异常检测
4. **建筑节能**: 建筑热桥检测
5. **科研应用**: 温度分布研究

### 部署环境
- **嵌入式系统**: 树莓派、Jetson Nano等
- **工业PC**: x86_64架构Linux系统
- **边缘计算设备**: ARM架构设备

## 🔍 扩展功能

### 即将实现
- [ ] 多线程优化（生产者-消费者模式）
- [ ] 内存池管理
- [ ] 红外特化算法优化
- [ ] Web控制面板
- [ ] 实时RTSP流媒体输出
- [ ] 云端数据同步

### 算法增强
- [ ] 深度学习边缘检测（YOLO检测+边缘提取）
- [ ] 自适应阈值调节
- [ ] 多光谱融合检测
- [ ] 历史数据对比分析

## 📞 故障排除

### 常见问题
1. **相机连接失败**
   - 检查相机驱动是否正确安装
   - 验证相机设备权限 (`ls -l /dev/video*`)
   - 使用 `v4l2-ctl --list-devices` 检查设备

2. **显示问题**
   - 使用无头模式运行: `python demo_headless.py`
   - 检查OpenCV Qt插件安装

3. **性能问题**
   - 调低分辨率和帧率
   - 禁用部分预处理步骤
   - 使用性能监控查看瓶颈

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统功能！

## 📄 许可证

MIT License - 详见LICENSE文件