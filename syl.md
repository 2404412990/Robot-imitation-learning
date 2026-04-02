# 2026/3/26
苏钰林跑通从视频到mujoco中h1动作重定向的流程

做了一个basketballFOX的测试样例

Current Limitation：

1、动作映射仍然存在卡顿现象，这与一开始GVHMR中视频映射到3dpose时数据不完整有关。

2、暂时的物理模型效果较差。

Future work：对视频中的动作状态进行补全。

# 2026/4/3
对GVHMR进行实时化改造

1、VideoCapture(0)	支持摄像头输入
  --skip_frames	跳帧处理，减轻推理压力
  window_size 8	从 10 降到 8，减少初始缓冲延迟
  --no_render	可选跳过 mesh 渲染，延迟减半
  cv2.imshow	实时显示窗口，按 q 退出
  --video 测试模式	无显示器时用视频文件验证流程
  ViTPose-B 权重（vitpose-b-multi-coco.pth）尚未获取，换上后 ViTPose 推理速度预计提升 3-4x

2、在服务器上用 kunkundance.mp4 跑通全流程，248帧无报错，稳定 ~2.5fps
