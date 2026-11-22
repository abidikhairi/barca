from pathlib import Path

from datasets import Dataset, DatasetDict

from barca.rl.data import build_grpo_dataset, parse_chat_row, select_records

REFERENCE = "MPQCKTSPLSNWHTFLFEYKVYFLEDMSVENQMYHVSRTKCVADPAYS"


def chat_row(user: str, reference: str) -> str:
    reply = "Here is the design:\n<protein>" + "".join("ζ" + aa for aa in reference) + "</protein>"
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{reply}<|im_end|>"
    )


def test_parse_chat_row() -> None:
    user, sequence = parse_chat_row(chat_row("Design a protein.", REFERENCE))
    assert user == "Design a protein."
    assert sequence == REFERENCE
    assert parse_chat_row("not a chat row") is None


def test_select_records_stratifies_lengths() -> None:
    records = [
        {"ref_len": 40, "value": index}
        for index in range(20)
    ] + [
        {"ref_len": 150, "value": 100 + index}
        for index in range(20)
    ] + [
        {"ref_len": 350, "value": 200 + index}
        for index in range(20)
    ]
    selected = select_records(records, num_prompts=6, seed=0, min_len=30, max_len=512)
    assert len(selected) == 6
    buckets = {0 if row["ref_len"] <= 100 else 1 if row["ref_len"] <= 200 else 2 for row in selected}
    assert buckets == {0, 1, 2}
    assert select_records(records, 6, 0, 30, 512) == selected


def test_build_grpo_dataset(tmp_path: Path) -> None:
    train = Dataset.from_list(
        [
            {"accession": "A1", "text": chat_row("First instruction", REFERENCE)},
            {"accession": "A2", "text": chat_row("Second instruction", REFERENCE)},
            {"accession": "A3", "text": chat_row("Validation instruction", REFERENCE)},
            {"accession": "A4", "text": chat_row("Third instruction", REFERENCE)},
            {"accession": "A5", "text": chat_row("Short instruction", "A" * 10)},
        ]
    )
    validation = Dataset.from_list(
        [{"accession": "A3", "text": chat_row("Validation instruction", REFERENCE)}]
    )
    data_path = tmp_path / "mol-instruct"
    DatasetDict({"train": train, "validation": validation}).save_to_disk(str(data_path))
    cache_dir = tmp_path / "cache"

    dataset = build_grpo_dataset(
        data_path=str(data_path),
        num_prompts=2,
        min_len=30,
        max_len=512,
        cache_dir=str(cache_dir),
    )
    assert len(dataset) == 2
    assert dataset.column_names == ["prompt", "reference", "ref_len", "accession", "idx"]
    assert "A3" not in dataset["accession"]
    assert all(row[0]["role"] == "user" for row in dataset["prompt"])
    assert all(row == REFERENCE for row in dataset["reference"])

    cached = build_grpo_dataset(
        data_path=str(data_path),
        num_prompts=2,
        min_len=30,
        max_len=512,
        cache_dir=str(cache_dir),
    )
    assert len(cached) == 2
