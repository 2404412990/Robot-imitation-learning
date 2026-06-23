# 2026.6.19

增加一个新分支，将漫游改成固定摄像机下的ui界面

<img width="1754" height="933" alt="2150505fcbc017574c42170e6bcf2130 (1)" src="https://github.com/user-attachments/assets/9db2d970-2859-4708-9bbb-2c8cc75e9bda" />


# 2026.6.5

1. 对于dataset下的所有离线csv，将它们重新放在dataset下面以机器人名字为名字的文件夹下管理，每次切换机器人的时候，csv下拉列表只会显示该机器人适配的csv文件

2. 解决unity实时重定向中机器人一直卡顿的问题：实际上是两者帧率不匹配：通常是unity下的机器人读取实时csv文件新行的速度快于wham_gmr产生新行的速度，从而导致untiy下的机器人需要经常去等数据，看起来有卡顿地效果，解决办法就是让快的去主动适应慢的，对齐两端的帧率

3. 增加对openloong和x02lite的支持，修复部分关节错位的问题，经测试g1, h1, openloong, x02lite（关节映射仍有一些问题）可以正常在unity内实时重定向

# 2026.5.8

适配windows系统，最主要是dpvo（slam估计）和pytorch3d（wham窗口渲染时的那个小人）的编译，然后也改了下windows下的格物平台，现在windows的格物平台和ubuntu 格物平台的功能差不多，我在windows下加个两个Raw image组件，可以抓取wham和gmr两个窗口并输出到unity世界内

现在还存在一些bug，比如在unity窗口内重定向过程中会非常卡，视角乱飞

之后的工作：格物平台可以再加一点新东西，可以再对wham和gmr进行一些修改，提升效果



https://github.com/user-attachments/assets/87b28df1-bd3f-4556-831c-10a8a914bc81





# 2026.5.1
解决了wham_gmr里gmr重定向后的机器人一直停在原地的问题，因为wham输出的是累加的全局根方向和全局根坐标，gmr本身不知道这是累加值，导致全局坐标没法累计

把wham_gmr完整串起来，之前wham和gmr是有两个wham和gmr的conda环境，而且它们之间交流也是通过磁盘IO。我把它们集成为一个conda环境，直接在内存内进行这两个流程


# 2026.4.24
稍微修复了一下ubuntu内格物平台imitation的一些bug


# 2026.4.17
格物平台集成，replay按钮以及旁边的动作csv下拉列表可以进行离线csv重定向操作，start按钮可以开始实时重定向,stop按钮终止重定向，下面两个层级表可以调整属性



https://github.com/user-attachments/assets/d5c057e6-5253-42d9-9c71-3027380cb572



https://github.com/user-attachments/assets/eec3391d-6220-4771-a2b4-1c8a6c9096e3




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




