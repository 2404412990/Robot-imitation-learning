# 2026.3.22:

<a href="https://github.com/zju3dv/GVHMR">GVHMR</a>

<a href="https://github.com/YanjieZe/GMR">GMR</a>

<a href="https://github.com/loongOpen/Unity-RL-Playground">格物平台</a>

用GVHMR把视频转换成human motion，然后用GMR把human motion重定向到g1机器人上,<br>
在学院服务器上成功复现，并放入格物平台复现

### 演示：

原视频输入：

<img src=docs/video/kunkundance.gif alt="animated" />

GVHMR转换：

<img src=docs/video/1_incam.gif alt="animated" /> <img src=docs/video/2_global.gif alt="animated" />

GMR转换：

<img src=docs/video/unitree_g1_hmr4d_results.gif alt="animated" />

导入至格物平台训练两千万轮：

<img src=docs/video/results.gif alt="animated" />
