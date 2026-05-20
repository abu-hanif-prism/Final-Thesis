from pathlib import Path

import pytest

from src.data_inventory.parse_filenames import (
    build_image_id,
    normalize_district_name,
    normalize_season_name,
    parse_satellite_filename,
)


def test_valid_sentinel_filename():
    result = parse_satellite_filename("S2_30m_2016_Winter_Khulna.tif")

    assert result["parse_status"] == "success"
    assert result["filename"] == "S2_30m_2016_Winter_Khulna.tif"
    assert result["stem"] == "S2_30m_2016_Winter_Khulna"
    assert result["sensor"] == "S2"
    assert result["resolution"] == "30m"
    assert result["year"] == 2016
    assert result["season"] == "Winter"
    assert result["district"] == "Khulna"
    assert result["image_id"] == "Khulna_2016_Winter"
    assert result["extension"] == ".tif"
    assert result["error_message"] is None


def test_valid_dynamic_world_filename():
    result = parse_satellite_filename(Path("DW_30m_2016_PreMonsoon_Khulna.tiff"))

    assert result["parse_status"] == "success"
    assert result["sensor"] == "DW"
    assert result["season"] == "PreMonsoon"
    assert result["district"] == "Khulna"
    assert result["image_id"] == "Khulna_2016_PreMonsoon"
    assert result["extension"] == ".tiff"


@pytest.mark.parametrize(
    ("filename", "season", "image_id"),
    [
        (
            "DW_allclasses_30m_2016_PreMonsoon_Satkhira.tif",
            "PreMonsoon",
            "Satkhira_2016_PreMonsoon",
        ),
        (
            "DW_allclasses_30m_2016_PostMonsoon_Satkhira.tif",
            "PostMonsoon",
            "Satkhira_2016_PostMonsoon",
        ),
        (
            "DW_allclasses_30m_2016_Monsoon_Satkhira.tif",
            "Monsoon",
            "Satkhira_2016_Monsoon",
        ),
        (
            "DW_allclasses_30m_2016_Winter_Satkhira.tif",
            "Winter",
            "Satkhira_2016_Winter",
        ),
    ],
)
def test_dynamic_world_allclasses_filename(filename, season, image_id):
    result = parse_satellite_filename(filename)

    assert result["parse_status"] == "success"
    assert result["sensor"] == "DW"
    assert result["resolution"] == "30m"
    assert result["year"] == 2016
    assert result["season"] == season
    assert result["district"] == "Satkhira"
    assert result["image_id"] == image_id
    assert result["extension"] == ".tif"
    assert result["error_message"] is None


def test_district_with_underscore():
    result = parse_satellite_filename("Sentinel2_30m_2017_Monsoon_Coxs_Bazar.tif")

    assert result["parse_status"] == "success"
    assert result["sensor"] == "Sentinel2"
    assert result["district"] == "Coxs_Bazar"
    assert result["image_id"] == "Coxs_Bazar_2017_Monsoon"


def test_district_with_spaces():
    result = parse_satellite_filename("DynamicWorld_30m_2018_PostMonsoon_Coxs Bazar.tif")

    assert result["parse_status"] == "success"
    assert result["sensor"] == "DynamicWorld"
    assert result["district"] == "Coxs_Bazar"
    assert result["season"] == "PostMonsoon"


def test_invalid_season():
    result = parse_satellite_filename("S2_30m_2019_Summer_Khulna.tif")

    assert result["parse_status"] == "failed"
    assert result["error_message"] is not None
    assert "Invalid season" in result["error_message"]


def test_invalid_year():
    result = parse_satellite_filename("S2_30m_2015_Winter_Khulna.tif")

    assert result["parse_status"] == "failed"
    assert result["error_message"] is not None
    assert "2016 to 2025" in result["error_message"]


def test_invalid_extension():
    result = parse_satellite_filename("S2_30m_2016_Winter_Khulna.jpg")

    assert result["parse_status"] == "failed"
    assert result["extension"] == ".jpg"
    assert result["error_message"] is not None
    assert "Invalid extension" in result["error_message"]


def test_completely_invalid_filename():
    result = parse_satellite_filename("not_a_valid_filename.tif")

    assert result["parse_status"] == "failed"
    assert result["sensor"] is None
    assert result["image_id"] is None
    assert result["error_message"] is not None


def test_lowercase_season_handling():
    result = parse_satellite_filename("Sentinel_30m_2020_premonsoon_Dhaka.tif")

    assert result["parse_status"] == "success"
    assert result["sensor"] == "Sentinel"
    assert result["season"] == "PreMonsoon"
    assert result["image_id"] == "Dhaka_2020_PreMonsoon"
    assert normalize_season_name("post_monsoon") == "PostMonsoon"


def test_lowercase_district_handling():
    result = parse_satellite_filename("S2_30m_2021_monsoon_coxs_bazar.tif")

    assert result["parse_status"] == "success"
    assert result["district"] == "Coxs_Bazar"
    assert normalize_district_name("coxs_bazar") == "Coxs_Bazar"
    assert build_image_id("coxs bazar", 2021, "monsoon") == "Coxs_Bazar_2021_Monsoon"


def test_invalid_normalize_season_raises_clear_error():
    with pytest.raises(ValueError, match="Invalid season"):
        normalize_season_name("Summer")
