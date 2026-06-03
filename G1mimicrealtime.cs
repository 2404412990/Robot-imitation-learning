using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections.Generic; // 解决 List<> 报错
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System;

public class G1mimicAgent_RealTime : Agent
{
    public bool train = false;
    public bool replay = false;

    float[] uff = new float[29];
    float[] u = new float[29];
    float[] utotal = new float[29];

    Transform body;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[29];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;

    [Header("实时通信设置")]
    public int udpPort = 5005;
    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning = true;

    // 线程安全的共享数据
    private float[] realtimeData = new float[36];
    private object lockObj = new object();
    private bool hasNewData = false;

    // 当前解析的数据
    float[] currentData = new float[36];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[29];

    public override void Initialize()
    {
        // 1. 初始化关节
        arts = this.GetComponentsInChildren<ArticulationBody>();
        int ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if (arts[k].jointType.ToString() == "RevoluteJoint")
            {
                jh[ActionNum] = arts[k];
                ActionNum++;
            }
        }
        body = jh[14].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        pos0 = body.position;
        rot0 = body.rotation;

        arts[0].immovable = false; // 确保获取时是35维，修复 ArticulationBody 缓存报错
        art0.GetJointPositions(P0);
        art0.GetJointVelocities(W0);

        // 2. 开启 UDP 监听线程
        udpClient = new UdpClient(udpPort);
        receiveThread = new Thread(ReceiveData) { IsBackground = true };
        receiveThread.Start();
        print("G1 UDP Receiver Started on port: " + udpPort);
    }

    private void ReceiveData()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        while (isRunning)
        {
            try
            {
                byte[] bytes = udpClient.Receive(ref anyIP);
                // 36 个 float = 36 * 4 = 144 字节
                if (bytes.Length == 144)
                {
                    lock (lockObj)
                    {
                        Buffer.BlockCopy(bytes, 0, realtimeData, 0, 144);
                        hasNewData = true;
                    }
                }
            }
            catch (Exception e) {
                Debug.LogError("Error receiving data: " + e.Message);
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        arts[0].immovable = false;
        arts[0].TeleportRoot(pos0, rot0);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;
        arts[0].SetJointPositions(P0);
        arts[0].SetJointVelocities(W0);
        for (int i = 0; i < 29; i++) u[i] = 0;
        for (int i = 0; i < 29; i++) uff[i] = 0;
        tt = 0;
    }

    void FixedUpdate()
    {
        // 1. 从 UDP 获取最新的一帧 36 维数据
        lock (lockObj)
        {
            if (hasNewData)
            {
                Array.Copy(realtimeData, currentData, 36);
                hasNewData = false;
            }
        }

        // 2. 切片解析数据
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 29);

        for (int i = 0; i < 29; i++)
            uff[i] = currentDof[i] * 180f / Mathf.PI; // 弧度转角度

        // 3. 坐标系转换 (根据你原来的逻辑)
        Quaternion gymQuat = new Quaternion(currentRot[0], currentRot[1], currentRot[2], currentRot[3]);
        Quaternion conversionQ = new Quaternion(0.5f, -0.5f, -0.5f, 0.5f);
        newRotation = conversionQ * gymQuat * Quaternion.Inverse(conversionQ);

        newPosition = new Vector3(-currentPos[1], currentPos[2] + 0.04f, currentPos[0]);
        newRotation = new Quaternion(currentRot[1], -currentRot[2], currentRot[0], currentRot[3]);

        newPosition.x += pos0.x;
        newPosition.z += pos0.z;

        // ==========================================
        // 纯模仿模式 (Replay/Teleoperation)
        // ==========================================
        if (replay)
        {
            Physics.gravity = Vector3.zero;
            arts[0].TeleportRoot(newPosition, newRotation);

            // 实时赋予各个关节目标角度
            for (int i = 0; i < 29; i++)
            {
                SetJointTargetDeg(jh[i], uff[i]);
            }
        }

        // ML-Agents 逻辑
        tt++;
        var live_reward = 1f;
        float rot_reward = 0;
        float pos_reward = 0;

        if (tt > 3)
        {
            arts[0].immovable = false;
            rot_reward = -0.01f * Quaternion.Angle(body.rotation, newRotation);
            pos_reward = -1f * (body.position - newPosition).magnitude;

            if (Quaternion.Angle(body.rotation, newRotation) > 30f || (body.position - newPosition).magnitude > 0.3f)
            {
                if (!replay) EndEpisode();
            }
        }

        var reward = live_reward + (rot_reward + pos_reward) * 1f;
        AddReward(reward);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * Mathf.PI / 180f);
        sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * Mathf.PI / 180f);
        sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        for (int i = 0; i < 29; i++)
        {
            sensor.AddObservation(jh[i].jointPosition[0]);
            sensor.AddObservation(jh[i].jointVelocity[0]);
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 20;
        if (replay) kb = 0;
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            if (i >= 15) kb = 0;
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.target = x;
        joint.xDrive = drive;
    }

    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180) return eulerAngle;
        else return eulerAngle - 360f;
    }

    // 去掉 override，因为 Agent 不是继承自具有 virtual OnDestroy 的父类
    void OnDestroy()
    {
        isRunning = false;
        if (udpClient != null) udpClient.Close();
        if (receiveThread != null) receiveThread.Abort();
    }
}
