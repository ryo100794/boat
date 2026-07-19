from pathlib import Path

from boatrace_ai.ingestion.archives import extract_lzh


def test_extract_lzh_reads_official_archive_without_context_manager() -> None:
    archive = Path("data/raw/result/2022/20220609.lzh")
    if not archive.exists():
        return

    members = extract_lzh(archive)

    assert len(members) == 1
    assert members[0][0] == "K220609.TXT"
    assert members[0][1].startswith(b"STARTK")
