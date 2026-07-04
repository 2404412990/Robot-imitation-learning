using System.Collections.Generic;

namespace Gewu.Imitation
{
    internal static class RobotCatalog
    {
        public const string UnitreeG1 = "unitree_g1";
        public const string UnitreeG1WithHands = "unitree_g1_with_hands";
        public const string UnitreeH1 = "unitree_h1";
        public const string UnitreeH12 = "unitree_h1_2";
        public const string X02Lite = "x02lite";
        public const string OpenLoong = "openloong";
        public const string Lite11V1 = "lite_11_v1";

        public static readonly string[] PrimaryDisplayOrder =
        {
            UnitreeG1,
            UnitreeH1,
            X02Lite,
            OpenLoong,
        };

        public static readonly string[] SupportedPipelineKeys =
        {
            UnitreeG1,
            UnitreeG1WithHands,
            UnitreeH1,
            UnitreeH12,
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            X02Lite,
            OpenLoong,
            Lite11V1,
            "tienkung",
            "fourier_gr3",
        };

        private static readonly Dictionary<string, string> Aliases =
            new Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase)
            {
                { "G1", UnitreeG1 },
                { "G1H", UnitreeG1WithHands },
                { "H1", UnitreeH1 },
                { "H1_2", UnitreeH12 },
                { "X02", X02Lite },
                { "X02Lite", X02Lite },
                { "OpenLoong", OpenLoong },
                { "Loong", OpenLoong },
                { "lite11", Lite11V1 },
                { "lite_11", Lite11V1 },
                { "lite-11", Lite11V1 },
                { "Lite_11_v1", Lite11V1 },
                { "lite-11-v1", Lite11V1 },
                { "T1", "booster_t1" },
                { "fourier_gr3v2_1_1", "fourier_gr3" },
            };

        private static readonly Dictionary<string, string> DatasetFolders =
            new Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase)
            {
                { UnitreeG1, "unitree_g1" },
                { UnitreeG1WithHands, "unitree_g1" },
                { UnitreeH1, "unitree_h1" },
                { UnitreeH12, "unitree_h1" },
                { X02Lite, "x02lite" },
                { OpenLoong, "openloong" },
                { Lite11V1, "lite_11_v1" },
            };

        private static readonly Dictionary<string, int> ExpectedCsvColumns =
            new Dictionary<string, int>(System.StringComparer.OrdinalIgnoreCase)
            {
                { UnitreeG1, 36 },
                { UnitreeG1WithHands, 36 },
                { UnitreeH1, 26 },
                { UnitreeH12, 26 },
                { X02Lite, 25 },
                { OpenLoong, 38 },
                { Lite11V1, 19 },
            };

        public static bool TryNormalizeKey(string rawRobotName, out string normalizedKey)
        {
            normalizedKey = string.Empty;
            if (string.IsNullOrWhiteSpace(rawRobotName))
            {
                return false;
            }

            string trimmed = rawRobotName.Trim();
            if (Aliases.TryGetValue(trimmed, out string aliasTarget))
            {
                normalizedKey = aliasTarget;
                return IsSupportedPipelineRobot(aliasTarget);
            }

            if (IsSupportedPipelineRobot(trimmed))
            {
                normalizedKey = trimmed;
                return true;
            }

            return false;
        }

        public static string NormalizeKeyOrOriginal(string rawRobotName)
        {
            if (string.IsNullOrWhiteSpace(rawRobotName))
            {
                return string.Empty;
            }

            return TryNormalizeKey(rawRobotName, out string normalizedKey)
                ? normalizedKey
                : rawRobotName.Trim();
        }

        public static bool IsSupportedPipelineRobot(string robotKey)
        {
            if (string.IsNullOrWhiteSpace(robotKey))
            {
                return false;
            }

            string trimmed = robotKey.Trim();
            for (int i = 0; i < SupportedPipelineKeys.Length; i++)
            {
                if (string.Equals(SupportedPipelineKeys[i], trimmed, System.StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        public static bool TryGetDatasetFolder(string robotKeyOrLabel, out string folder)
        {
            folder = string.Empty;
            string normalized = NormalizeKeyOrOriginal(robotKeyOrLabel);
            return DatasetFolders.TryGetValue(normalized, out folder);
        }

        public static string GetDatasetFolderOrKey(string robotKeyOrLabel)
        {
            return TryGetDatasetFolder(robotKeyOrLabel, out string folder)
                ? folder
                : (robotKeyOrLabel ?? string.Empty).Trim();
        }

        public static bool TryGetExpectedCsvColumns(string robotKeyOrLabel, out int expectedColumns)
        {
            expectedColumns = 0;
            string normalized = NormalizeKeyOrOriginal(robotKeyOrLabel);
            return ExpectedCsvColumns.TryGetValue(normalized, out expectedColumns);
        }
    }
}
