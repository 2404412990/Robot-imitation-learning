using System;
using UnityEngine;

namespace Gewu.Imitation
{
    [Serializable]
    public struct UnityRetargetCalibrationEntry
    {
        public int csvIndex;
        public string jointName;
        public float sign;
        public float offsetRad;
        public bool clampToDriveLimits;

        public UnityRetargetCalibrationEntry(int csvIndex, string jointName, float sign, float offsetRad, bool clampToDriveLimits = true)
        {
            this.csvIndex = csvIndex;
            this.jointName = jointName;
            this.sign = Mathf.Approximately(sign, 0f) ? 1f : Mathf.Sign(sign);
            this.offsetRad = offsetRad;
            this.clampToDriveLimits = clampToDriveLimits;
        }

        public float Apply(float csvRad)
        {
            float safeSign = Mathf.Approximately(sign, 0f) ? 1f : Mathf.Sign(sign);
            return safeSign * csvRad + offsetRad;
        }
    }

    public static class UnityRetargetCalibration
    {
        public static UnityRetargetCalibrationEntry Resolve(
            int csvIndex,
            string jointName,
            UnityRetargetCalibrationEntry[] table,
            float defaultSign = 1f,
            float defaultOffsetRad = 0f,
            bool defaultClampToDriveLimits = true)
        {
            if (table != null)
            {
                for (int i = 0; i < table.Length; i++)
                {
                    if (table[i].csvIndex == csvIndex)
                    {
                        return table[i];
                    }
                }
            }

            return new UnityRetargetCalibrationEntry(
                csvIndex,
                jointName,
                Mathf.Approximately(defaultSign, 0f) ? 1f : defaultSign,
                defaultOffsetRad,
                defaultClampToDriveLimits);
        }
    }

    public static class UnityQposMapper
    {
        public static Vector3 MapRootPosition(float[] qposRootPosition, Vector3 initialUnityRoot, Vector3 displayOffset)
        {
            if (qposRootPosition == null || qposRootPosition.Length < 3)
            {
                return initialUnityRoot + displayOffset;
            }

            Vector3 mapped = new Vector3(-qposRootPosition[1], qposRootPosition[2], qposRootPosition[0]);
            mapped.x += initialUnityRoot.x;
            mapped.z += initialUnityRoot.z;
            return mapped + displayOffset;
        }

        public static Quaternion MapRootRotationFromCsvXyzw(float[] qposRootRotationXyzw)
        {
            if (qposRootRotationXyzw == null || qposRootRotationXyzw.Length < 4)
            {
                return Quaternion.identity;
            }

            // GMR writes MuJoCo qpos as CSV xyzw. This is the Unity mapping
            // already validated by the G1 path: (x,y,z,w) -> (-y,z,x,-w).
            Quaternion mapped = new Quaternion(
                -qposRootRotationXyzw[1],
                 qposRootRotationXyzw[2],
                 qposRootRotationXyzw[0],
                -qposRootRotationXyzw[3]);
            return Normalize(mapped);
        }

        public static Quaternion Normalize(Quaternion rotation)
        {
            float magnitude = Mathf.Sqrt(
                rotation.x * rotation.x +
                rotation.y * rotation.y +
                rotation.z * rotation.z +
                rotation.w * rotation.w);
            if (magnitude < 0.0001f)
            {
                return Quaternion.identity;
            }

            float inv = 1f / magnitude;
            return new Quaternion(rotation.x * inv, rotation.y * inv, rotation.z * inv, rotation.w * inv);
        }

        public static bool TrySetJointPositionRad(ArticulationBody joint, float radians, bool zeroVelocity)
        {
            if (joint == null || joint.jointType != ArticulationJointType.RevoluteJoint)
            {
                return false;
            }

            ArticulationReducedSpace jointPosition = joint.jointPosition;
            if (jointPosition.dofCount <= 0)
            {
                return false;
            }

            jointPosition[0] = radians;
            joint.jointPosition = jointPosition;

            if (zeroVelocity)
            {
                ArticulationReducedSpace jointVelocity = joint.jointVelocity;
                if (jointVelocity.dofCount > 0)
                {
                    jointVelocity[0] = 0f;
                    joint.jointVelocity = jointVelocity;
                }
            }

            return true;
        }
    }
}
