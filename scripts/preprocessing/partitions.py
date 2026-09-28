"""Fixed source-dataset partitions used by the preprocessing loader.

These lists are intentionally checked into source control.  Do not derive them
from the filesystem at runtime: changing a partition is a deliberate dataset
contract change that should be visible in review.
"""

from __future__ import annotations

from collections.abc import Sequence


# Original 133 assignments are unchanged. Twelve complete extracted3 sources
# were assigned 8/2/2 with a fixed shuffle (seed 20260928).
TRAIN_IDS = [
    "0005+383",
    "0006-063",
    "0012-399",
    "0024-420",
    "0025-260",
    "0029+349",
    "0042+233",
    "0059+001",
    "0112+227",
    "0116-116",
    "0129+236",
    "0132-169",
    "0141+138",
    "0145-275",
    "0149+059",
    "0153-331",
    "0201-115",
    "0203+115",
    "0204+152",
    "0205+322",
    "0240-231",
    "0242-215",
    "0259+077",
    "0312-148",
    "0321+123",
    "0323+055",
    "0329+279",
    "0403+260",
    "0405-131",
    "0409-179",
    "0416-188",
    "0416-209",
    "0432+416",
    "0449+113",
    "0508+845",
    "0539-286",
    "0608-223",
    "0609-157",
    "0616-349",
    "0650-166",
    "0653+370",
    "0653-064",
    "0735+331",
    "0738+177",
    "0739+016",
    "0744-064",
    "0745+101",
    "0805+617",
    "0818+423",
    "0832+492",
    "0834+555",
    "0841+708",
    "0846-261",
    "0921+622",
    "0925+003",
    "0948+406",
    "0954+177",
    "0956+252",
    "0958+474",
    "1016+206",
    "1018-317",
    "1024-008",
    "1033+395",
    "1033+412",
    "1044+809",
    "1048+717",
    "1111+199",
    "1119-030",
    "1125+261",
    "1130+382",
    "1147-074",
    "1147-382",
    "1150+242",
    "1159+292",
    "1209-241",
    "1215+348",
    "1221+282",
    "1224+035",
    "1224+213",
    "1246-075",
    "1248-199",
    "1305-105",
    "1309+119",
    "1327+221",
    "1349+536",
    "1352-442",
    "1354-021",
    "1411+522",
    "1415+133",
    "1416+347",
    "1430+107",
    "1432-180",
    "1436+233",
    "1439-169",
    "1448-163",
    "1500+478",
    "1504+104",
    "1505+034",
    "1510-057",
    "1513+236",
    "1923-210",
]

TEST_IDS = [
    "1146+399",
    "1310+323",
    "1513-102",
    "1520+202",
    "1522-275",
    "1549+506",
    "1557-000",
    "1602+334",
    "1609+266",
    "1613+342",
    "1617+027",
    "1625+415",
    "1634+627",
    "1635+381",
    "1640+123",
    "1653+397",
    "1719+177",
    "1743-038",
    "1824+107",
    "1911-201",
    "1924+334",
    "1924-292",
]

VAL_IDS = [
    "0022+002",
    "0725-009",
    "1927+612",
    "1949-199",
    "2007+404",
    "2011-067",
    "2023+544",
    "2040-251",
    "2137+510",
    "2202+422",
    "2212+018",
    "2218-035",
    "2241+098",
    "2246-121",
    "2248-325",
    "2257-364",
    "2316+040",
    "2330+110",
    "2333+390",
    "2333-237",
    "2341-351",
    "2357-114",
]

PARTITION_IDS: dict[str, list[str]] = {
    "train": TRAIN_IDS,
    "test": TEST_IDS,
    "val": VAL_IDS,
}
PARTITION_NAMES = tuple(PARTITION_IDS)
ALL_IDS = tuple(dataset_id for ids in PARTITION_IDS.values() for dataset_id in ids)


def _validate_partitions(partitions: Sequence[Sequence[str]]) -> None:
    flattened = [dataset_id for partition in partitions for dataset_id in partition]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("Preprocessing partitions contain duplicate dataset IDs")


_validate_partitions(tuple(PARTITION_IDS.values()))


def normalize_partition(partition: str) -> str:
    """Return the canonical partition name or reject an unsupported value."""

    if not isinstance(partition, str):
        raise TypeError("partition must be a string")
    normalized = partition.strip().casefold()
    if normalized not in PARTITION_IDS:
        choices = ", ".join(PARTITION_NAMES)
        raise ValueError(f"Unknown partition {partition!r}; expected one of: {choices}")
    return normalized


def get_partition_ids(partition: str) -> tuple[str, ...]:
    """Return an immutable view of the IDs assigned to ``partition``."""

    return tuple(PARTITION_IDS[normalize_partition(partition)])


def source_dataset_id(sample_id: str) -> str:
    """Resolve a retained sample/variant ID to its source dataset ID.

    A base sample such as ``0012-399`` and variants such as
    ``0012-399_phase_only`` remain in the same partition.
    """

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    matches = [
        dataset_id
        for dataset_id in ALL_IDS
        if sample_id == dataset_id or sample_id.startswith(f"{dataset_id}_")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Sample ID {sample_id!r} does not match exactly one partitioned dataset ID"
        )
    return matches[0]


def partition_for_sample(sample_id: str) -> str:
    """Return the partition containing ``sample_id`` and all its variants."""

    dataset_id = source_dataset_id(sample_id)
    return next(
        name for name, dataset_ids in PARTITION_IDS.items() if dataset_id in dataset_ids
    )


__all__ = [
    "ALL_IDS",
    "PARTITION_IDS",
    "PARTITION_NAMES",
    "TEST_IDS",
    "TRAIN_IDS",
    "VAL_IDS",
    "get_partition_ids",
    "normalize_partition",
    "partition_for_sample",
    "source_dataset_id",
]
