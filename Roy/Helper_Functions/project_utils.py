import re
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES_DIR = PROJECT_ROOT / "Test_Images"
V2_DIR = PROJECT_ROOT / "Roy" / "V2"
S2_INDEX_PATH = V2_DIR / "s2_index.pt"

_TEST_IMAGE_RE = re.compile(r"(.+)_Test(\d+)$")


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def get_test_images_dir() -> Path:
    return TEST_IMAGES_DIR


def get_s2_index_path() -> Path:
    return S2_INDEX_PATH


def parse_test_image(path_or_name: str) -> Tuple[str, int]:
    stem = Path(path_or_name).stem
    match = _TEST_IMAGE_RE.fullmatch(stem)
    if not match:
        raise ValueError(path_or_name)
    return match.group(1), int(match.group(2)) - 1


def list_test_image_series(test_images_dir: Optional[Path] = None) -> List[str]:
    folder = test_images_dir or TEST_IMAGES_DIR
    series = set()

    if not folder.exists():
        return []

    for item in folder.iterdir():
        if not item.is_file():
            continue
        try:
            series.add(parse_test_image(item.name)[0])
        except ValueError:
            continue

    return sorted(series)


def get_test_image_path(test_type: str, image_idx: int, test_images_dir: Optional[Path] = None) -> Optional[Path]:
    folder = test_images_dir or TEST_IMAGES_DIR
    if not folder.exists():
        return None

    matches = sorted(folder.glob(f"{test_type}_Test{image_idx + 1}.*"))
    if not matches:
        return None

    return matches[0]
