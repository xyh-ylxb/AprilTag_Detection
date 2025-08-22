#include "Eigen/Dense"
#include <iostream>
#include <cmath>

namespace vita_slam {
namespace vs_base {
// 常量定义
const float kDeg2RadF = M_PI / 180.0f;

// common data type using in slam
template <typename T>
using Eig_Vec2 = Eigen::Vector2<T>;
template <typename T>
using Eig_Vec3 = Eigen::Vector3<T>;
template <typename T>
using Eig_Vec4 = Eigen::Vector4<T>;

template <typename T>
using Eig_Mat2 = Eigen::Matrix2<T>;
template <typename T>
using Eig_Mat3 = Eigen::Matrix3<T>;
template <typename T>
using Eig_Mat4 = Eigen::Matrix4<T>;

// float类型的类型别名
using Eig_Vec2f = Eig_Vec2<float>;
using Eig_Vec3f = Eig_Vec3<float>;
using Eig_Vec4f = Eig_Vec4<float>;
using Eig_Mat2f = Eig_Mat2<float>;
using Eig_Mat3f = Eig_Mat3<float>;
using Eig_Mat4f = Eig_Mat4<float>;

template <typename T>
Eig_Mat3<T> IntrXYZ2R(const Eig_Vec3<T>& xyz) {
  // 分别提取绕X、Y、Z轴的旋转角度
  T roll = xyz(0);   // X轴
  T pitch = xyz(1);  // Y轴
  T yaw = xyz(2);    // Z轴

  // 计算各轴旋转的三角函数值
  T cx = std::cos(roll);
  T sx = std::sin(roll);
  T cy = std::cos(pitch);
  T sy = std::sin(pitch);
  T cz = std::cos(yaw);
  T sz = std::sin(yaw);

  Eig_Mat3<T> Rx;
  Rx << 1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx;
  Eig_Mat3<T> Ry;
  Ry << cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy;
  Eig_Mat3<T> Rz;
  Rz << cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0;

  return Rx * Ry * Rz;
}

template <typename T>
Eig_Vec3<T> IntrR2XYZ(const Eig_Mat3<T>& R) {
  // pitch
  T pitch =
      std::atan2(R(0, 2), std::sqrt(R(0, 0) * R(0, 0) + R(0, 1) * R(0, 1)));

  T roll;
  T yaw;
  if (std::abs(R(0, 2)) > 0.9998) {
    // gimbal lock
  } else {
    roll = std::atan2(-R(1, 2), R(2, 2));
    yaw = std::atan2(-R(0, 1), R(0, 0));
  }

  Eig_Vec3<T> rpy(roll, pitch, yaw);
  return rpy;
}
}
}

int main() {
  // 原始坐标变换演示
  std::cout << "=== 原始坐标变换演示 ===" << std::endl;
  vita_slam::vs_base::Eig_Vec3f rpy(10.f, 10.f, 10.f);
  rpy *= vita_slam::vs_base::kDeg2RadF;

  vita_slam::vs_base::Eig_Mat3f R_b_i = vita_slam::vs_base::IntrXYZ2R(rpy);

  std::cout << "rpy_b_i:\n" << rpy.transpose() << std::endl;
  std::cout << "R_b_i:\n" << R_b_i << std::endl;

  vita_slam::vs_base::Eig_Vec3f imu_rpy(4.f, 0.f, 0.f);
  imu_rpy *= vita_slam::vs_base::kDeg2RadF;
  std::cout << "imu_rpy(rad):\n" << imu_rpy.transpose() << std::endl;
  vita_slam::vs_base::Eig_Mat3f R_body =
      R_b_i * vita_slam::vs_base::IntrXYZ2R(imu_rpy) * R_b_i.transpose();
  vita_slam::vs_base::Eig_Vec3f body_rpy =
      vita_slam::vs_base::IntrR2XYZ(R_body);
  std::cout << "body_rpy(rad):\n" << body_rpy.transpose() << std::endl;
  
  // ========== 正确的IMU安装偏差模拟 ==========
  std::cout << "\n=== 正确的IMU安装偏差模拟 ===" << std::endl;
  
  // 1. 实际的安装偏差（我们不知道这个）
  vita_slam::vs_base::Eig_Vec3f true_install_error(10.f, 10.f, 10.f);
  true_install_error *= vita_slam::vs_base::kDeg2RadF;
  vita_slam::vs_base::Eig_Mat3f R_install_error = 
      vita_slam::vs_base::IntrXYZ2R(true_install_error);
  
  // 2. 机器人body系实际发生的旋转（4度roll）
  vita_slam::vs_base::Eig_Vec3f true_body_rotation(4.f, 0.f, 0.f);
  true_body_rotation *= vita_slam::vs_base::kDeg2RadF;
  vita_slam::vs_base::Eig_Mat3f R_true_body = 
      vita_slam::vs_base::IntrXYZ2R(true_body_rotation);
  
  // 3. IMU实际测量到的旋转（由于安装偏差，不是4度）
  // IMU测量 = 安装偏差的逆 × body旋转 × 安装偏差
  vita_slam::vs_base::Eig_Mat3f R_imu_measurement = 
      R_install_error.transpose() * R_true_body * R_install_error;
  vita_slam::vs_base::Eig_Vec3f imu_measured_rpy = 
      vita_slam::vs_base::IntrR2XYZ(R_imu_measurement);
  
  // 4. 如果我们不知道安装偏差，错误地认为IMU和body系重合
  // 那么我们会认为body系也发生了IMU测量的旋转
  vita_slam::vs_base::Eig_Mat3f R_wrong_body = R_imu_measurement;
  vita_slam::vs_base::Eig_Vec3f wrong_body_rpy = 
      vita_slam::vs_base::IntrR2XYZ(R_wrong_body);
  
  // 输出结果
  std::cout << "实际安装偏差 (度): " << (true_install_error * 180.0f / M_PI).transpose() << std::endl;
  std::cout << "机器人body系实际旋转 (度): " << (true_body_rotation * 180.0f / M_PI).transpose() << std::endl;
  std::cout << "IMU实际测量到的旋转 (度): " << (imu_measured_rpy * 180.0f / M_PI).transpose() << std::endl;
  std::cout << "错误认为的body系旋转 (度): " << (wrong_body_rpy * 180.0f / M_PI).transpose() << std::endl;
  
  // 计算误差
  vita_slam::vs_base::Eig_Vec3f measurement_error = 
      wrong_body_rpy - true_body_rotation;
  std::cout << "测量误差 (度): " << (measurement_error * 180.0f / M_PI).transpose() << std::endl;

  // ========== 反问题：IMU读到4度，body实际旋转多少度 ==========
  std::cout << "\n=== 反问题测试 ===" << std::endl;
  // 1. 已知IMU测量
  vita_slam::vs_base::Eig_Vec3f imu_reading(4.f, 0.f, 0.f);
  imu_reading *= vita_slam::vs_base::kDeg2RadF;
  vita_slam::vs_base::Eig_Mat3f R_imu_measured = vita_slam::vs_base::IntrXYZ2R(imu_reading);

  // 2. 已知安装偏差
  // 已有 R_install_error

  // 3. 反推body实际旋转
  vita_slam::vs_base::Eig_Mat3f R_body_actual = R_install_error * R_imu_measured * R_install_error.transpose();
  vita_slam::vs_base::Eig_Vec3f body_actual_rpy = vita_slam::vs_base::IntrR2XYZ(R_body_actual);

  std::cout << "IMU测量 (度): " << (imu_reading * 180.0f / M_PI).transpose() << std::endl;
  std::cout << "反推body实际旋转 (度): " << (body_actual_rpy * 180.0f / M_PI).transpose() << std::endl;
  
  // ========== 加速度和角速度模拟 ==========
  std::cout << "\n=== 加速度和角速度模拟 ===" << std::endl;
  // 1. 设定body系下的真实角速度和加速度
  vita_slam::vs_base::Eig_Vec3f omega_body(0.1f, 0.2f, 0.3f); // rad/s
  vita_slam::vs_base::Eig_Vec3f acc_body(1.0f, 0.0f, 0.0f);   // m/s^2

  // 2. 变换到IMU系
  vita_slam::vs_base::Eig_Vec3f omega_imu = R_install_error.transpose() * omega_body;
  vita_slam::vs_base::Eig_Vec3f acc_imu = R_install_error.transpose() * acc_body;

  std::cout << "body系真实角速度 (rad/s): " << omega_body.transpose() << std::endl;
  std::cout << "IMU测量角速度 (rad/s): " << omega_imu.transpose() << std::endl;
  std::cout << "body系真实加速度 (m/s^2): " << acc_body.transpose() << std::endl;
  std::cout << "IMU测量加速度 (m/s^2): " << acc_imu.transpose() << std::endl;

  // 3. 反推body系真实值（已知IMU测量和安装偏差）
  vita_slam::vs_base::Eig_Vec3f omega_body_recover = R_install_error * omega_imu;
  vita_slam::vs_base::Eig_Vec3f acc_body_recover = R_install_error * acc_imu;

  std::cout << "反推body系角速度 (rad/s): " << omega_body_recover.transpose() << std::endl;
  std::cout << "反推body系加速度 (m/s^2): " << acc_body_recover.transpose() << std::endl;
  
  return 0;
}