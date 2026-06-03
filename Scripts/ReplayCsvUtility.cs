using System.Collections.Generic;
using UnityEngine;

public static class ReplayCsvUtility
{
    public const float SourceFps = 30f;
    public const float UnityFixedFps = 50f;

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
            result.Add(row);
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

        return row;
    }
}
