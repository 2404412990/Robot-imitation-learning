# 2026.4.10
在本地安装ubuntu系统，将前一周的代码下载到本地运行

## 流程集成：
把WHAM的输出转换为smplx骨架格式，再输入到GMR，将这两个流程串起来

上一周我们已经实现了WHAM实时输入，这周我让WHAM的输出在一个文件中持续更新，GMR持续追踪更新的文件，实现整个流程的实时化

## 性能优化：
将yolo和vitpose换成更轻量的模型，开启多线程优化，跳帧，半精度推理，减小时序窗口，缩放图像等方法在不太影响精度的情况下提高帧率

最终效果：在4060laptop下，wham和gmr基本可以同步，没有延迟，总fps在15帧左右，显存占用 5.6/8 G



https://github.com/user-attachments/assets/3e824556-8cfa-499a-981b-285b2dab85e8



# 2026.4.3
由于GVHMR实时处理时帧率较低，故考虑优化

对于GVHMR，用轻量vitpose-b替换原有vitpose-h，给realtime.py加上多线程处理以及跳帧处理，即便优化后在3090 GPU上帧率也只维持在1~2左右

<img src=docs/img/GVHMR_REALTIME_OUTPUT.png />

因为GVHMR的主体网络用的是transformer，对实时输入不太擅长，故换为主体网络为RNN的<a href="https://github.com/yohanshin/WHAM">WHAM</a>

在仅加入多线程优化的realtime.py下，虽然精度不如GVHMR的transformer，但帧率可以稳定在7.8~9 fps之间。

<img src=docs/video/WHAM_REALTIME.gif alt="animated" />


<img src=docs/video/subject3.gif alt="animated" />


## Future works

搭建gewu平台离线输入视频和实时输入摄像头画面，接受服务器传回数据的前端框架

尝试将平台从3090换至A100

在本地运行WHAM，测试真实摄像头下的表现

探索更多优化方法

# 2026.3.27:

<a href="https://github.com/zju3dv/GVHMR">GVHMR</a>

<a href="https://github.com/YanjieZe/GMR">GMR</a>

<a href="https://github.com/loongOpen/Unity-RL-Playground">格物平台</a>

用GVHMR把视频转换成human motion，然后用GMR把human motion重定向到g1机器人上,<br>
在学院服务器上成功复现，并放入格物平台运行

### 演示：

原视频输入：

<img src=docs/video/kunkundance.gif alt="animated" />

GVHMR转换：

<img src=docs/video/1_incam.gif alt="animated" /> <img src=docs/video/2_global.gif alt="animated" />

GMR转换：

<img src=docs/video/unitree_g1_hmr4d_results.gif alt="animated" />

导入至格物平台训练两千万轮：

<img src=docs/video/results.gif alt="animated" />




