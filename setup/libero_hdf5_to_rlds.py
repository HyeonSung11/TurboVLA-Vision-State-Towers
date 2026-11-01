#!/usr/bin/env python3
"""LIBERO no-noops HDF5 -> TFDS/RLDS 변환기.

TurboVLA 저장소는 이 변환 단계를 제공하지 않습니다 (README: "then convert the
derived trajectories to TFDS/RLDS"). 아래 스펙은 학습 로더 코드를 읽어 확정한 것입니다.

로더가 요구하는 것 (turbovla/data/libero_rlds.py):
    episode["steps"][t]["observation"]["image"]        (256,256,3) uint8
    episode["steps"][t]["observation"]["wrist_image"]  (256,256,3) uint8
    episode["steps"][t]["observation"]["state"]        (8,)  float32
    episode["steps"][t]["action"]                      (7,)  float32
    episode["steps"][t]["language_instruction"]        string

  - _ensure_expected_size(): 로더는 리사이즈를 하지 않으며 정확히 256x256 을 요구합니다
    (trainer.py --expected_image_size 기본값 256).
  - state 8-D = ee_states(6: eef_pos 3 + axis-angle 3) + gripper_states(2).
    libero_all4_stats.json 의 proprio mean/std 가 8차원인 것과 일치합니다.
  - action 7-D.

이미지 회전 (중요):
  평가는 두 뷰 모두에 rotate_libero_image() = img[::-1, ::-1] (180도) 를 적용합니다
  (turbovla/evaluation/policy.py:167, 546-547). 반면 regenerate_libero_no_noops.py 는
  회전하지 않은 원본을 HDF5 에 저장하고, 학습 로더도 회전하지 않습니다.
  따라서 학습/평가 입력을 맞추려면 이 변환기가 회전을 적용해야 합니다.
  (--no_rotate 로 끌 수 있지만 권장하지 않습니다.)

사용:
  python libero_hdf5_to_rlds.py \
      --suite libero_object \
      --hdf5_dir ~/code/turbovla/LIBERO/libero/datasets/libero_object_no_noops \
      --out_root ~/code/turbovla/TurboVLA/data/libero
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds
from etils import epath

IMAGE_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 7

# TurboVLA 가 기대하는 출력 디렉토리 이름 (README 의 data/libero/ 레이아웃)
SUITE_TO_DIRNAME = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}


def build_language_map(suite: str) -> dict[str, str]:
    """LIBERO benchmark 에서 task.name -> task.language 매핑을 만듭니다."""
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite]()
    return {
        task_suite.get_task(i).name: task_suite.get_task(i).language
        for i in range(task_suite.n_tasks)
    }


def rotate(img: np.ndarray) -> np.ndarray:
    """평가측 rotate_libero_image() 와 동일한 180도 회전."""
    return np.ascontiguousarray(img[::-1, ::-1])


def iter_episodes(hdf5_dir: Path, lang_map: dict[str, str], do_rotate: bool):
    """no-noops HDF5 파일들을 순회하며 RLDS 에피소드를 만들어 냅니다."""
    files = sorted(hdf5_dir.glob("*_demo.hdf5"))
    if not files:
        raise FileNotFoundError(f"{hdf5_dir} 에 *_demo.hdf5 파일이 없습니다")

    for path in files:
        task_name = path.name[: -len("_demo.hdf5")]
        instruction = lang_map.get(task_name)
        if instruction is None:
            # benchmark 에 없으면 파일명에서 복원 (SCENE 접두사 제거)
            instruction = task_name.split("_SCENE", 1)[-1].lstrip("0123456789_").replace("_", " ")
            print(f"  [warn] '{task_name}' 의 language 를 benchmark 에서 못 찾음 -> '{instruction}'")

        with h5py.File(path, "r") as f:
            data = f["data"]
            for demo_key in sorted(data.keys(), key=lambda k: int(k.split("_")[1])):
                demo = data[demo_key]
                obs = demo["obs"]

                agent = obs["agentview_rgb"][()]
                wrist = obs["eye_in_hand_rgb"][()]
                ee = obs["ee_states"][()]
                grip = obs["gripper_states"][()]
                actions = demo["actions"][()]

                length = len(actions)
                assert len(agent) == length, f"{path}:{demo_key} 길이 불일치"

                state = np.concatenate(
                    [ee[:length, :6], grip[:length, :2]], axis=1
                ).astype(np.float32)
                assert state.shape[1] == STATE_DIM, f"state dim {state.shape[1]} != {STATE_DIM}"
                assert actions.shape[1] == ACTION_DIM, f"action dim {actions.shape[1]} != {ACTION_DIM}"

                steps = []
                for t in range(length):
                    img = agent[t]
                    wimg = wrist[t]
                    if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError(
                            f"{path}:{demo_key} 이미지가 {img.shape[:2]} 입니다. "
                            f"{IMAGE_SIZE}x{IMAGE_SIZE} 여야 합니다 "
                            f"(regenerate 스크립트의 IMAGE_RESOLUTION=256 확인)"
                        )
                    if do_rotate:
                        img, wimg = rotate(img), rotate(wimg)

                    steps.append({
                        "observation": {
                            "image": img.astype(np.uint8),
                            "wrist_image": wimg.astype(np.uint8),
                            "state": state[t],
                        },
                        "action": actions[t].astype(np.float32),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if t == length - 1 else 0.0),
                        "is_first": t == 0,
                        "is_last": t == length - 1,
                        "is_terminal": t == length - 1,
                        "language_instruction": instruction,
                    })

                yield f"{path.stem}/{demo_key}", {
                    "steps": steps,
                    "episode_metadata": {"file_path": str(path)},
                }


def make_builder_cls(hdf5_dir: Path, lang_map: dict, do_rotate: bool, encoding: str):
    class LiberoNoNoops(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        RELEASE_NOTES = {"1.0.0": "LIBERO no-noops -> RLDS for TurboVLA."}
        # 이 클래스는 스크립트 실행 시 __main__ 에 정의되므로 TFDS 의
        # 패키지 경로 추론(_get_builder_datadir_path)이 실패합니다.
        # 메타데이터 디렉토리를 직접 지정해 그 경로를 건너뜁니다.
        pkg_dir_path = epath.Path(str(Path(__file__).parent / "_tfds_meta"))

        def _info(self):
            return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict({
                    "steps": tfds.features.Dataset({
                        "observation": tfds.features.FeaturesDict({
                            "image": tfds.features.Image(
                                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                dtype=np.uint8, encoding_format=encoding,
                            ),
                            "wrist_image": tfds.features.Image(
                                shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                dtype=np.uint8, encoding_format=encoding,
                            ),
                            "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                        }),
                        "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                        "discount": tfds.features.Scalar(dtype=np.float32),
                        "reward": tfds.features.Scalar(dtype=np.float32),
                        "is_first": tfds.features.Scalar(dtype=np.bool_),
                        "is_last": tfds.features.Scalar(dtype=np.bool_),
                        "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                        "language_instruction": tfds.features.Text(),
                    }),
                    "episode_metadata": tfds.features.FeaturesDict({
                        "file_path": tfds.features.Text(),
                    }),
                }),
            )

        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}

        def _generate_examples(self):
            yield from iter_episodes(hdf5_dir, lang_map, do_rotate)

    return LiberoNoNoops


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True, choices=sorted(SUITE_TO_DIRNAME))
    p.add_argument("--hdf5_dir", required=True,
                   help="regenerate_libero_no_noops.py 의 --libero_target_dir")
    p.add_argument("--out_root", required=True,
                   help="예: <TurboVLA>/data/libero")
    p.add_argument("--image_encoding", default="jpeg", choices=["jpeg", "png"],
                   help="jpeg=작음(손실), png=무손실(수배 큼). 기본 jpeg")
    p.add_argument("--no_rotate", action="store_true",
                   help="180도 회전을 끕니다. 평가와 불일치하므로 권장하지 않습니다")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    hdf5_dir = Path(args.hdf5_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    target = out_root / SUITE_TO_DIRNAME[args.suite] / "1.0.0"

    if target.exists():
        if not args.overwrite:
            raise SystemExit(f"이미 존재합니다: {target}  (--overwrite 로 덮어쓰기)")
        shutil.rmtree(target)

    print(f"suite   : {args.suite}")
    print(f"입력    : {hdf5_dir}")
    print(f"출력    : {target}")
    print(f"회전    : {'적용 안 함 (비권장)' if args.no_rotate else '180도 적용 (평가와 일치)'}")
    print(f"인코딩  : {args.image_encoding}")

    lang_map = build_language_map(args.suite)
    print(f"task 수 : {len(lang_map)}")

    cls = make_builder_cls(hdf5_dir, lang_map, not args.no_rotate, args.image_encoding)

    with tempfile.TemporaryDirectory(dir=str(out_root.parent)) as tmp:
        builder = cls(data_dir=tmp)
        builder.download_and_prepare()

        built = Path(tmp) / "libero_no_noops" / "1.0.0"
        if not built.exists():
            cands = list(Path(tmp).glob("*/1.0.0"))
            if len(cands) != 1:
                raise SystemExit(f"빌드 결과 디렉토리를 찾지 못했습니다: {list(Path(tmp).rglob('*'))[:20]}")
            built = cands[0]

        # dataset_info.json 의 name 과 tfrecord 파일명을 함께 바꿉니다.
        # 이름만 바꾸면 리더가 <new_name>-train.tfrecord-* 를 찾는데 실제
        # 파일은 <old_name>-train.tfrecord-* 이라 NotFoundError 가 납니다.
        target_name = SUITE_TO_DIRNAME[args.suite]
        info_path = built / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text())
            old_name = info.get("name", "")
            info["name"] = target_name
            info_path.write_text(json.dumps(info, indent=2))
            if old_name and old_name != target_name:
                for f in built.iterdir():
                    if f.name.startswith(f"{old_name}-"):
                        f.rename(f.with_name(target_name + f.name[len(old_name):]))

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built), str(target))

    print(f"\n완료: {target}")
    print(f"크기: {sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
