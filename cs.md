# 4.17 进行方法探索
我使用RTMpose3d进行姿态映射到GMR转换为机器人的实时映射，但是效果很差，可能是由于RTMpose3d输出的不是smpl参数，而是17个关键点，并且缺少人体姿态估计，直接将17个点转为机器人的关节，因此效果没有WHAM+GMR的好，但是速度却比较快，并且延迟很低。下周计划采用[TRACE](https://github.com/Arthur151/ROMP)，这个模型可以直接输出SMPL的 24 个高精度三维物理关节，外加真实的根节点全局旋转角，并且自带时序平滑，可以减少动作的畸变。

# 4.3 进行方法探索和试错
我们先是将GVHMR的输入转化为一帧一帧的视频来模仿实时输入，并且维护一个10帧大小的窗口来输入给模型进行姿态提取，目前速度较慢，一秒处理一帧，还没有算上GMR进行重映射的时间，所以放弃GVHMR模型改用更加轻量的[WHAM](https://github.com/yohanshin/WHAM)模型；此外我们又尝试了[mediapipe](https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6)进行直接端到端动作映射，但其由于是单眼摄像头映射，所以精度很差，虽然帧率较高也不会考虑。使用WHAM模型帧率达到了10帧左右，虽然较慢，但还有优化空间，可以使用[RTMpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)替代姿态提取模型进一步提高速度。

## 不足
这个流程虽然相较于GVHMR较快，但是由于还要进行人体动作的重映射，再次将WHAM的输出结果传递给GMR输出为动作序列文件来进行replay，所以整个流程比较割裂，还需切换环境，可以做进一步优化集成。

# 3.27 视频到机器人动作映射 (GVHMR + GMR)
使用**GVHMR**从视频中提取人体姿态，然后通过 **GMR** 将人体运动重映射为机器人关节动作，最终生成 **CSV** 格式的动作序列文件，可直接用于机器人控制或仿真。
![Video Project](https://github.com/user-attachments/assets/f8971420-dc46-4a50-a730-338e8b8655df)

## 不足
目前这个流程虽然可以高精度的处理从视频到机器人的动作映射，但是这个系统仍是非实时性，处理速度慢，生成的 CSV 只能用于开环播放，无法根据机器人反馈动态调整动作，难以应用于需要实时交互的场景。

是否可以在自己电脑上部署一个YOLOv8并通过TCP通信来实现实时的动作映射
