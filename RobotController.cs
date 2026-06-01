using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class G1Teleoperation : MonoBehaviour
{
    [Header("UDP Receiver")]
    public int port = 5005;
    private UdpClient udpClient;
    private Thread receiveThread;
    private Vector3[] mpNodes = new Vector3[33];
    private bool dataUpdated = false;
    private object lockObj = new object();

    [Header("G1 Torso & Waist (3 DOF)")]
    public Transform waist_yaw_link;
    public Transform waist_roll_link;
    public Transform torso_link; // waist_pitch

    [Header("G1 Left Leg (6 DOF)")]
    public Transform left_hip_pitch_link;
    public Transform left_hip_roll_link;
    public Transform left_hip_yaw_link;
    public Transform left_knee_link;
    public Transform left_ankle_pitch_link;
    public Transform left_ankle_roll_link;

    [Header("G1 Left Arm (7 DOF)")]
    public Transform left_shoulder_pitch_link;
    public Transform left_shoulder_roll_link;
    public Transform left_shoulder_yaw_link;
    public Transform left_elbow_link;
    public Transform left_wrist_pitch_link;
    public Transform left_wrist_roll_link;
    public Transform left_wrist_yaw_link;

    [Header("G1 Right Leg (6 DOF)")]
    public Transform right_hip_pitch_link;
    public Transform right_hip_roll_link;
    public Transform right_hip_yaw_link;
    public Transform right_knee_link;
    public Transform right_ankle_pitch_link;
    public Transform right_ankle_roll_link;

    [Header("G1 Right Arm (7 DOF)")]
    public Transform right_shoulder_pitch_link;
    public Transform right_shoulder_roll_link;
    public Transform right_shoulder_yaw_link;
    public Transform right_elbow_link;
    public Transform right_wrist_pitch_link;
    public Transform right_wrist_roll_link;
    public Transform right_wrist_yaw_link;

    // --- 省略右手和右腿的 public 声明，与左侧对称 ---

    [System.Serializable] public class Landmark { public float x; public float y; public float z; public float v; }
    [System.Serializable] public class LandmarkData { public Landmark[] landmarks; }

    void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(ReceiveData) { IsBackground = true };
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                string json = Encoding.UTF8.GetString(udpClient.Receive(ref anyIP));
                LandmarkData data = JsonUtility.FromJson<LandmarkData>(json);

                if (data != null && data.landmarks.Length == 33)
                {
                    lock (lockObj)
                    {
                        for (int i = 0; i < 33; i++)
                        {
                            // MediaPipe 到 Unity 坐标系转换 (镜像映射)
                            mpNodes[i] = new Vector3(-data.landmarks[i].x, -data.landmarks[i].y, data.landmarks[i].z);
                        }
                        dataUpdated = true;
                    }
                }
            }
            catch { }
        }
    }

    void Update()
    {
        lock (lockObj)
        {
            if (!dataUpdated) return;

            // ==========================================
            // 1. 获取关键点 (以人体盆骨中心为原点)
            // ==========================================
            Vector3 hipCenter = (mpNodes[23] + mpNodes[24]) / 2f;
            Vector3 shoulderCenter = (mpNodes[11] + mpNodes[12]) / 2f;

            // --- 左半边 ---
            Vector3 lShoulder = mpNodes[11];
            Vector3 lElbow = mpNodes[13];
            Vector3 lWrist = mpNodes[15];

            Vector3 lHip = mpNodes[23];
            Vector3 lKnee = mpNodes[25];
            Vector3 lAnkle = mpNodes[27];

            Vector3 rShoulder = mpNodes[12];
            Vector3 rElbow = mpNodes[14];
            Vector3 rWrist = mpNodes[16];

            Vector3 rHip = mpNodes[24];
            Vector3 rKnee = mpNodes[26];
            Vector3 rAnkle = mpNodes[28];

            // ==========================================
            // 2. 几何角度反解 (根据 XML 中的 axis 定义)
            // ==========================================

            // 【腰部 3 DOF】(Waist Yaw/Roll/Pitch)
            Vector3 spineVec = (shoulderCenter - hipCenter).normalized;
            // 投影到各个面计算角度
            float waistYaw = Mathf.Atan2(spineVec.x, spineVec.z) * Mathf.Rad2Deg;
            float waistPitch = Mathf.Atan2(spineVec.z, spineVec.y) * Mathf.Rad2Deg;
            float waistRoll = Mathf.Atan2(spineVec.x, spineVec.y) * Mathf.Rad2Deg;

            if (waist_yaw_link) waist_yaw_link.localEulerAngles = new Vector3(0, 0, waistYaw); // Z轴
            if (waist_roll_link) waist_roll_link.localEulerAngles = new Vector3(waistRoll, 0, 0); // X轴
            if (torso_link) torso_link.localEulerAngles = new Vector3(0, waistPitch, 0); // Y轴

            // 【左臂 Shoulder 3 DOF + Elbow 1 DOF】
            Vector3 lUpperArm = (lElbow - lShoulder).normalized;
            Vector3 lLowerArm = (lWrist - lElbow).normalized;

            // 肩膀 Pitch (Y轴, 矢状面), Roll (X轴, 冠状面)
            float lShoulderPitch = Mathf.Atan2(lUpperArm.z, -lUpperArm.y) * Mathf.Rad2Deg;
            float lShoulderRoll = Mathf.Atan2(lUpperArm.x, -lUpperArm.y) * Mathf.Rad2Deg;

            // 肘部 Pitch (Y轴) - 通过大臂和小臂夹角计算
            float lElbowPitch = Vector3.Angle(lUpperArm, lLowerArm);

            if (left_shoulder_pitch_link) left_shoulder_pitch_link.localEulerAngles = new Vector3(0, lShoulderPitch, 0);
            if (left_shoulder_roll_link) left_shoulder_roll_link.localEulerAngles = new Vector3(lShoulderRoll, 0, 0);
            if (left_elbow_link) left_elbow_link.localEulerAngles = new Vector3(0, lElbowPitch, 0);

            // 【左腿 Hip 3 DOF + Knee 1 DOF】
            Vector3 lThigh = (lKnee - lHip).normalized;
            Vector3 lCalf = (lAnkle - lKnee).normalized;

            float lHipPitch = Mathf.Atan2(lThigh.z, -lThigh.y) * Mathf.Rad2Deg;
            float lHipRoll = Mathf.Atan2(lThigh.x, -lThigh.y) * Mathf.Rad2Deg;
            float lKneePitch = Vector3.Angle(lThigh, lCalf); // 膝盖弯曲角

            if (left_hip_pitch_link) left_hip_pitch_link.localEulerAngles = new Vector3(0, lHipPitch, 0);
            if (left_hip_roll_link) left_hip_roll_link.localEulerAngles = new Vector3(lHipRoll, 0, 0);
            if (left_knee_link) left_knee_link.localEulerAngles = new Vector3(0, lKneePitch, 0); // Y轴

            Vector3 rUpperArm = (rElbow - rShoulder).normalized;
            Vector3 rLowerArm = (rWrist - rElbow).normalized;

            // 肩膀 Pitch (Y轴, 矢状面), Roll (X轴, 冠状面)
            float rShoulderPitch = Mathf.Atan2(rUpperArm.z, -rUpperArm.y) * Mathf.Rad2Deg;
            float rShoulderRoll = Mathf.Atan2(rUpperArm.x, -rUpperArm.y) * Mathf.Rad2Deg;

            // 肘部 Pitch (Y轴) - 通过大臂和小臂夹角计算
            float rElbowPitch = Vector3.Angle(rUpperArm, rLowerArm);

            if (right_shoulder_pitch_link) right_shoulder_pitch_link.localEulerAngles = new Vector3(0, rShoulderPitch, 0);
            if (right_shoulder_roll_link) right_shoulder_roll_link.localEulerAngles = new Vector3(rShoulderRoll, 0, 0);
            if (right_elbow_link) right_elbow_link.localEulerAngles = new Vector3(0, rElbowPitch, 0);

            // 【左腿 Hip 3 DOF + Knee 1 DOF】
            Vector3 rThigh = (rKnee - rHip).normalized;
            Vector3 rCalf = (rAnkle - rKnee).normalized;

            float rHipPitch = Mathf.Atan2(rThigh.z, -rThigh.y) * Mathf.Rad2Deg;
            float rHipRoll = Mathf.Atan2(rThigh.x, -rThigh.y) * Mathf.Rad2Deg;
            float rKneePitch = Vector3.Angle(rThigh, rCalf); // 膝盖弯曲角

            if (right_hip_pitch_link) right_hip_pitch_link.localEulerAngles = new Vector3(0, rHipPitch, 0);
            if (right_hip_roll_link) right_hip_roll_link.localEulerAngles = new Vector3(rHipRoll, 0, 0);
            if (right_knee_link) right_knee_link.localEulerAngles = new Vector3(0, rKneePitch, 0); // Y轴

            dataUpdated = false;
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null) receiveThread.Abort();
        if (udpClient != null) udpClient.Close();
    }
}
