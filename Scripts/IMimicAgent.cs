using System.Collections.Generic;
using UnityEngine;

namespace Gewu.Imitation
{
    /// <summary>
    /// Common abstraction for any humanoid imitation agent that can be driven by
    /// the WHAM + GMR live retargeting pipeline (or by a pre-recorded CSV in
    /// replay mode). Different robot bodies (Unitree G1, Unitree H1, ...) all
    /// implement this interface so the UI scripts (StartInput / Stop / Replay)
    /// don't need to know which concrete agent type they are talking to.
    /// </summary>
    public interface IMimicAgent
    {
        /// <summary>
        /// Robot key used by the WHAM + GMR pipeline (e.g. "unitree_g1",
        /// "unitree_h1"). Must match one of the supported names in
        /// StartInput.SupportedRobotNames so the pipeline can be launched
        /// against this agent.
        /// </summary>
        string RobotKey { get; }

        /// <summary>
        /// GameObject this agent lives on — used by the registry / UI scripts
        /// to enable / disable / move the matching scene object.
        /// </summary>
        GameObject AgentGameObject { get; }

        /// <summary>
        /// Whether the agent is currently consuming external (live) replay
        /// data. When true, OnEpisodeBegin should NOT reload motion data
        /// from the on-disk dataset.
        /// </summary>
        bool UseExternalReplayData { get; set; }

        /// <summary>
        /// Whether the agent is in replay (kinematic) mode rather than
        /// PD-tracking training mode.
        /// </summary>
        bool ReplayMode { get; set; }

        /// <summary>
        /// Index of the currently selected motion clip in the on-disk
        /// dataset (used only when not in live mode).
        /// </summary>
        int MotionId { get; set; }

        /// <summary>
        /// Load a CSV file from disk and apply it as the current replay
        /// reference trajectory. Returns true if the load succeeded.
        /// </summary>
        bool LoadReplayCsvFromPath(string filePath, bool keepProgress);

        /// <summary>
        /// Request that ML-Agents end the current episode (so a fresh
        /// trajectory starts from the first frame on the next physics step).
        /// </summary>
        void RequestEndEpisode();

        /// <summary>
        /// Imperatively reset the agent's articulation back to its spawn
        /// pose: root teleported to the initial position/rotation captured at
        /// Initialize(), every revolute joint snapped back to its rest cache
        /// values, all xDrive.target values zeroed, and internal counters
        /// cleared. Unlike RequestEndEpisode (which is async — it merely queues
        /// OnEpisodeBegin for the next ML-Agents step), this method takes
        /// effect immediately and works even when the GameObject is inactive.
        ///
        /// Use case: dropdown robot switching. After SetActive(true) on the
        /// incoming robot, calling ResetToInitialState clears stale xDrive
        /// targets left over from before the previous deactivation, so the
        /// one-frame transient between activation and OnEpisodeBegin doesn't
        /// snap joints to wrong positions.
        /// </summary>
        void ResetToInitialState();
    }

    /// <summary>
    /// Optional realtime CSV ingestion surface used by StartInput. Unlike
    /// LoadReplayCsvFromPath, this appends only newly-written CSV rows so live
    /// playback does not periodically reparse and restart the whole file.
    /// </summary>
    public interface IRealtimeCsvMimicAgent
    {
        int ExpectedCsvColumns { get; }
        bool BeginRealtimeCsv();
        void SetRealtimePlaybackRate(float framesPerSecond, float bufferSeconds);
        bool AppendRealtimeCsvRows(IReadOnlyList<float[]> rows);
        void EndRealtimeCsv();
    }

    /// <summary>
    /// Optional visibility/selection hook used by StartInput when robots are
    /// hidden via renderer toggles instead of SetActive(false). Implementations
    /// can freeze physics and suppress policy/replay updates while hidden.
    /// </summary>
    public interface ISelectableMimicAgent
    {
        void SetRobotSelectedInScene(bool isSelected);
    }
}
