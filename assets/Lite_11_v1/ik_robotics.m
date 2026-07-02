% 加载必要的工具箱
% addpath(fullfile(matlabroot, 'toolbox', 'robotics'));
close all
clear all

% Step 1: Load the URDF file into a rigidBodyTree object
robot = importrobot('./urdf/miniloong_v4_12dof_c_new_V51.urdf'); % 替换为你的URDF文件路径
robot.DataFormat = 'column';

% 显示机器人的连杆和关节信息（可选）
showdetails(robot);
initialGuess = [0.25;0;0;-0.75;-0.5;0;-0.25;0;0;-0.75;-0.5;0];
figure
show(robot,initialGuess);
title('Initial guess')

% robot.JointPositions =initialGuess;

leftFootEndEffectorName = 'link_l_ankle_roll'; % 左脚末端执行器名称
rightFootEndEffectorName = 'link_r_ankle_roll'; % 右脚末端执行器名称

% Step 4: Calculate the initial position of the foot using forward kinematics
% 获取左脚相对于基座的位置和姿态
T_left_foot = getTransform(robot,initialGuess, 'base_link', leftFootEndEffectorName);

% 获取右脚相对于基座的位置和姿态
T_right_foot = getTransform(robot, initialGuess,'base_link', rightFootEndEffectorName);

% 提取平移部分作为位置坐标
leftFootPosition = T_left_foot(1:3, 4)'
rightFootPosition = T_right_foot(1:3, 4)'

% Step 2: Define the desired end-effector pose based on the given height
desiredHeight = 0.4; % 给定的机身高度 (单位：米)

% 假设我们想要计算左腿的关节角度，且已知机身高度为 desiredHeight 米
footPosition = [0, 0, -desiredHeight]; % 脚的位置相对于髋关节（假设Z轴向下）

% % 假设我们知道期望的姿态（例如，直立站立时的姿态）
% footOrientation = eye(3); % 单位矩阵表示无旋转
% 
% % 将位置和姿态组合成齐次变换矩阵
% T_desired = axang2tform([0 0 1 0]); % 默认无旋转
% T_desired(1:3, 4) = footPosition; % 设置平移部分

% 使用 axang2rotm 函数从轴角创建旋转矩阵
rotationAngle = pi / 2; % 90 度的弧度值
axis = [0 0 1]; % 绕 Z 轴旋转
R_z_90 = axang2rotm([axis rotationAngle]);

% 创建旋转矩阵，使 X 轴指向正前方，Y 轴指向左侧，Z 轴指向上方
R = [1 0 0;
     0 1 0;
     0 0 1];

% 将位置和姿态组合成齐次变换矩阵
T_desired = [R, footPosition';
             0, 0, 0, 1];

% Step 3: Setup the inverse kinematics solver for left leg
ik_left = inverseKinematics('RigidBodyTree', robot);

% 指定末端执行器名称（需要根据你的机器人模型替换为实际的末端执行器名称）
endEffectorName_left = 'link_l_ankle_roll'; % 左脚末端执行器名称

% 设置求解器参数（可选）
ik_left.SolverParameters.AllowRandomRestart = true;
ik_left.SolverParameters.MaxIterations = 100;

% Step 4: Solve for joint angles for left leg

% 初始化猜测值，这里设置为零向量
% initialGuess_left = zeros(12, 1);
weights = ones(6, 1); % 权重向量，对于6DOF，每个自由度的权重相同

% 正确调用 inverseKinematics 求解器
[configSol_left, solInfo_left] = ik_left(endEffectorName_left, T_desired, weights, initialGuess);

% 检查是否找到了有效解
if solInfo_left.ExitFlag == 1
    disp('Solution found for left leg.');
else
    warning('No valid solution found for left leg.');
end

% 显示结果
disp('Calculated Joint Angles for Left Leg:');
disp(configSol_left);

% 可视化解决方案（可选）
figure;
show(robot, configSol_left);
title('Left Leg IK Solution');

% 假设我们想要计算左腿的关节角度，且已知机身高度为 desiredHeight 米
footPosition = [0, 0, -desiredHeight]; % 脚的位置相对于髋关节（假设Z轴向下）

% % 假设我们知道期望的姿态（例如，直立站立时的姿态）
% footOrientation = eye(3); % 单位矩阵表示无旋转
% 
% % 将位置和姿态组合成齐次变换矩阵
% T_desired = axang2tform([0 0 1 0]); % 默认无旋转
% T_desired(1:3, 4) = footPosition; % 设置平移部分

% 创建旋转矩阵，使 X 轴指向正前方，Y 轴指向左侧，Z 轴指向上方
R = [1 0 0;
     0 1 0;
     0 0 1];

% 将位置和姿态组合成齐次变换矩阵
T_desired = [R, footPosition';
             0, 0, 0, 1];


% Step 5: Repeat similar steps for right leg if necessary
ik_right = inverseKinematics('RigidBodyTree', robot);
endEffectorName_right = 'link_r_ankle_roll'; % 右脚末端执行器名称

% 初始化右腿的猜测值
% initialGuess_right = zeros(12,1);
% initialGuess_right = [0;0;0;0;0;0;0;-0.25;0;0;-0.75;-0.5];

% 设置相同的期望位置和姿态
[configSol_right, solInfo_right] = ik_right(endEffectorName_right, T_desired, weights, initialGuess);

% 检查是否找到了有效解
if solInfo_right.ExitFlag == 1
    disp('Solution found for right leg.');
else
    warning('No valid solution found for right leg.');
end

% 显示结果
disp('Calculated Joint Angles for Right Leg:');
disp(configSol_right);

% 可视化解决方案（可选）
figure;
show(robot, configSol_right);
title('Right Leg IK Solution');