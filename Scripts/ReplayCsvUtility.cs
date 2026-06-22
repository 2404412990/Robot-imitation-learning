using System.Collections.Generic;
using UnityEngine;

public static class ReplayCsvUtility
{
    public const float SourceFps = 30f;
    public const float UnityFixedFps = 50f;
    public const float MinRealtimeFps = 1f;
    public const float MaxRealtimeFps = 120f;

    public static List<float[]> Resample30FpsToFixed50Hz(List<float[]> source)
    {
        if (source == null || source.Count <= 1)
        {
            return source == null ? new List<float[]>() : new List<float[]>(source);
        }

        int dimension = source[0].Length;
        for (int i = 1; i < source.Count; i++)
        {
            if (source[i].Length != dimension)
            {
                Debug.LogError("[ReplayCsvUtility] Cannot resample rows with inconsistent column counts.");
                return new List<float[]>(source);
            }
        }

        int newFrameCount = Mathf.Max(1, (int)(source.Count * UnityFixedFps / SourceFps) - 1);
        var result = new List<float[]>(newFrameCount);

        for (int i = 0; i < newFrameCount; i++)
        {
            float sourceFrame = i * SourceFps / UnityFixedFps;
            int left = Mathf.Clamp(Mathf.FloorToInt(sourceFrame), 0, source.Count - 1);
            int right = Mathf.Clamp(left + 1, 0, source.Count - 1);
            float ratio = Mathf.Clamp01(sourceFrame - left);

            float[] row = new float[dimension];
            for (int j = 0; j < dimension; j++)
            {
                row[j] = Mathf.Lerp(source[left][j], source[right][j], ratio);
            }
            InterpolateRootQuaternion(source[left], source[right], ratio, row, dimension);
            result.Add(row);
        }

        return result;
    }

    public static List<float[]> CopyRows(IReadOnlyList<float[]> source, int expectedDimension)
    {
        var result = new List<float[]>();
        if (source == null)
        {
            return result;
        }

        int dimension = expectedDimension > 0
            ? expectedDimension
            : (source.Count > 0 && source[0] != null ? source[0].Length : 0);
        if (dimension <= 0)
        {
            return result;
        }

        for (int i = 0; i < source.Count; i++)
        {
            float[] row = source[i];
            if (row == null || row.Length < dimension)
            {
                Debug.LogWarning($"[ReplayCsvUtility] Skip replay row with invalid column count: {row?.Length ?? 0}, expected {dimension}.");
                continue;
            }

            float[] copy = new float[dimension];
            System.Array.Copy(row, 0, copy, 0, dimension);
            NormalizeRootQuaternion(copy, dimension);
            result.Add(copy);
        }

        return result;
    }

    public static int AppendResampled30FpsToFixed50Hz(
        List<float[]> source,
        List<float[]> target,
        IReadOnlyList<float[]> newSourceRows,
        int expectedDimension)
    {
        if (source == null || target == null || newSourceRows == null || newSourceRows.Count == 0)
        {
            return 0;
        }

        int dimension = expectedDimension > 0
            ? expectedDimension
            : (source.Count > 0 ? source[0].Length : newSourceRows[0].Length);

        for (int i = 0; i < newSourceRows.Count; i++)
        {
            float[] row = newSourceRows[i];
            if (row == null || row.Length < dimension)
            {
                Debug.LogWarning($"[ReplayCsvUtility] Skip realtime row with invalid column count: {row?.Length ?? 0}, expected {dimension}.");
                continue;
            }

            float[] copy = new float[dimension];
            System.Array.Copy(row, 0, copy, 0, dimension);
            source.Add(copy);
        }

        int oldTargetCount = target.Count;
        int targetFrameCount = GetResampledFrameCount30To50(source.Count);
        while (target.Count < targetFrameCount)
        {
            target.Add(SampleSourceAtFixedFrame(source, target.Count, dimension));
        }

        return target.Count - oldTargetCount;
    }

    public static int AppendRawRows(
        List<float[]> target,
        IReadOnlyList<float[]> newRows,
        int expectedDimension)
    {
        return AppendRawRows(target, newRows, expectedDimension, copyRows: true);
    }

    public static int AppendRawRows(
        List<float[]> target,
        IReadOnlyList<float[]> newRows,
        int expectedDimension,
        bool copyRows)
    {
        if (target == null || newRows == null || newRows.Count == 0)
        {
            return 0;
        }

        int appended = 0;
        for (int i = 0; i < newRows.Count; i++)
        {
            float[] row = newRows[i];
            if (row == null || row.Length < expectedDimension)
            {
                Debug.LogWarning($"[ReplayCsvUtility] Skip realtime row with invalid column count: {row?.Length ?? 0}, expected {expectedDimension}.");
                continue;
            }

            if (copyRows)
            {
                float[] copy = new float[expectedDimension];
                System.Array.Copy(row, 0, copy, 0, expectedDimension);
                target.Add(copy);
            }
            else
            {
                target.Add(row);
            }
            appended++;
        }

        return appended;
    }

    public static float ClampRealtimeFps(float framesPerSecond)
    {
        if (float.IsNaN(framesPerSecond) || float.IsInfinity(framesPerSecond) || framesPerSecond <= 0f)
        {
            return SourceFps;
        }

        return Mathf.Clamp(framesPerSecond, MinRealtimeFps, MaxRealtimeFps);
    }

    public static float[] SampleRowsAtFrame(IReadOnlyList<float[]> rows, float frameCursor, int expectedDimension)
    {
        int dimension = expectedDimension > 0
            ? expectedDimension
            : (rows != null && rows.Count > 0 && rows[0] != null ? rows[0].Length : 0);
        if (dimension <= 0)
        {
            return null;
        }

        float[] sampled = new float[dimension];
        return SampleRowsAtFrame(rows, frameCursor, dimension, sampled) ? sampled : null;
    }

    public static bool SampleRowsAtFrame(
        IReadOnlyList<float[]> rows,
        float frameCursor,
        int expectedDimension,
        float[] destination)
    {
        if (rows == null || rows.Count == 0 || destination == null)
        {
            return false;
        }

        int dimension = expectedDimension > 0 ? expectedDimension : rows[0]?.Length ?? 0;
        if (dimension <= 0 || destination.Length < dimension)
        {
            return false;
        }

        float clampedCursor = Mathf.Clamp(frameCursor, 0f, rows.Count - 1);
        int left = Mathf.Clamp(Mathf.FloorToInt(clampedCursor), 0, rows.Count - 1);
        int right = Mathf.Clamp(left + 1, 0, rows.Count - 1);
        float ratio = Mathf.Clamp01(clampedCursor - left);

        float[] leftRow = rows[left];
        float[] rightRow = rows[right];
        if (leftRow == null || rightRow == null || leftRow.Length < dimension || rightRow.Length < dimension)
        {
            return false;
        }

        for (int i = 0; i < dimension; i++)
        {
            destination[i] = Mathf.Lerp(leftRow[i], rightRow[i], ratio);
        }
        InterpolateRootQuaternion(leftRow, rightRow, ratio, destination, dimension);

        return true;
    }

    public static float AdvanceRealtimeCursor(
        float frameCursor,
        int availableFrameCount,
        float framesPerSecond,
        float deltaTime,
        float targetBufferSeconds = 0f)
    {
        if (availableFrameCount <= 0)
        {
            return 0f;
        }

        float absoluteMaxFrame = availableFrameCount - 1;
        frameCursor = Mathf.Clamp(frameCursor, 0f, absoluteMaxFrame);
        float fps = ClampRealtimeFps(framesPerSecond);
        int targetBufferFrames = Mathf.Max(0, Mathf.FloorToInt(fps * Mathf.Max(0f, targetBufferSeconds)));
        float maxPlayableFrame = absoluteMaxFrame <= targetBufferFrames
            ? absoluteMaxFrame
            : Mathf.Max(0f, absoluteMaxFrame - targetBufferFrames);
        if (frameCursor >= maxPlayableFrame)
        {
            return frameCursor;
        }

        return Mathf.Min(maxPlayableFrame, frameCursor + fps * Mathf.Max(0f, deltaTime));
    }

    public static int GetResampledFrameCount30To50(int sourceFrameCount)
    {
        if (sourceFrameCount <= 1)
        {
            return Mathf.Max(0, sourceFrameCount);
        }

        return Mathf.Max(1, (int)(sourceFrameCount * UnityFixedFps / SourceFps) - 1);
    }

    private static float[] SampleSourceAtFixedFrame(List<float[]> source, int fixedFrameIndex, int dimension)
    {
        float sourceFrame = fixedFrameIndex * SourceFps / UnityFixedFps;
        int left = Mathf.Clamp(Mathf.FloorToInt(sourceFrame), 0, source.Count - 1);
        int right = Mathf.Clamp(left + 1, 0, source.Count - 1);
        float ratio = Mathf.Clamp01(sourceFrame - left);

        float[] row = new float[dimension];
        for (int j = 0; j < dimension; j++)
        {
            row[j] = Mathf.Lerp(source[left][j], source[right][j], ratio);
        }
        InterpolateRootQuaternion(source[left], source[right], ratio, row, dimension);

        return row;
    }

    private static void InterpolateRootQuaternion(float[] leftRow, float[] rightRow, float ratio, float[] destination, int dimension)
    {
        if (leftRow == null || rightRow == null || destination == null || dimension < 7)
        {
            NormalizeRootQuaternion(destination, dimension);
            return;
        }

        float lx = leftRow[3];
        float ly = leftRow[4];
        float lz = leftRow[5];
        float lw = leftRow[6];
        float rx = rightRow[3];
        float ry = rightRow[4];
        float rz = rightRow[5];
        float rw = rightRow[6];

        float dot = lx * rx + ly * ry + lz * rz + lw * rw;
        if (dot < 0f)
        {
            rx = -rx;
            ry = -ry;
            rz = -rz;
            rw = -rw;
        }

        destination[3] = Mathf.Lerp(lx, rx, ratio);
        destination[4] = Mathf.Lerp(ly, ry, ratio);
        destination[5] = Mathf.Lerp(lz, rz, ratio);
        destination[6] = Mathf.Lerp(lw, rw, ratio);
        NormalizeRootQuaternion(destination, dimension);
    }

    private static void NormalizeRootQuaternion(float[] row, int dimension)
    {
        if (row == null || dimension < 7)
        {
            return;
        }

        float x = row[3];
        float y = row[4];
        float z = row[5];
        float w = row[6];
        float sqrMagnitude = x * x + y * y + z * z + w * w;
        if (float.IsNaN(sqrMagnitude) || float.IsInfinity(sqrMagnitude) || sqrMagnitude < 1e-8f)
        {
            row[3] = 0f;
            row[4] = 0f;
            row[5] = 0f;
            row[6] = 1f;
            return;
        }

        float invMagnitude = 1f / Mathf.Sqrt(sqrMagnitude);
        row[3] = x * invMagnitude;
        row[4] = y * invMagnitude;
        row[5] = z * invMagnitude;
        row[6] = w * invMagnitude;
    }
}
