# 2026.4.3
由于GVHMR实时处理时帧率较低，故考虑优化

对于GVHMR，用轻量vitpose-b替换原有vitpose-h，给realtime.py加上多线程处理以及跳帧处理，即便优化后在3090 GPU上帧率也只维持在1~2左右

<img src=docs/img/GVHMR_REALTIME_OUTPUT.png />

因为GVHMR的主体网络用的是transformer，对实时输入不太擅长，故换为主体网络为RNN的<a href="https://github.com/yohanshin/WHAM">WHAM</a>

在仅加入多线程优化的realtime.py下，虽然精度不如GVHMR的transformer，但帧率可以稳定在7.8~9 fps之间。

<img src=docs/video/WHAM_REALTIME.gif alt="animated" />

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




