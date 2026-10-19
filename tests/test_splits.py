import pandas as pd

from green_rtm.data.metadata import attach_source_fields, infer_source_fields
from green_rtm.data.splits import assert_no_emit_leak, emit_free_split, unlabeled_block_for_fold


def test_emit_inference():
    assert infer_source_fields("something_EMIT_scene")["is_emit"]
    assert infer_source_fields("PRISMA_italy")["sensor"] == "PRISMA"
    assert infer_source_fields("NEON_AOP_2018")["platform"] == "airborne"


def test_emit_free_no_leak():
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "dataset": ["AVIRIS-NG", "EMIT_africa", "ASD_field"],
            "LAI": [1.0, 2.0, 3.0],
        }
    )
    df = attach_source_fields(df)
    rec = emit_free_split(df)
    keep = df.loc[df["sample_id"].isin(rec["train_ids"])]
    assert_no_emit_leak(keep)
    assert "EMIT_africa" in rec["held_out_sources"]


def test_unlabeled_block():
    un = attach_source_fields(
        pd.DataFrame({"sample_id": ["u1", "u2"], "dataset": ["PRISMA", "AVIRIS-NG"]})
    )
    blk = unlabeled_block_for_fold(un, ["PRISMA"])
    assert "u1" in blk["blocked_ids"]
    assert "u2" in blk["pretrain_ids"]
