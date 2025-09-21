# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This script is used to generate customized splits for the
PointNav task with stretch robot configuration.
"""
import argparse
import glob
import gzip
import json
import multiprocessing
import os
from os import path as osp

import habitat
import numpy as np
import tqdm
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_sim.nav import NavMeshSettings

from embodied_splat.config import HabitatConfigPlugin
from embodied_splat.utils.pointnav_generator import generate_pointnav_episode

os.environ["GLOG_minloglevel"] = "2"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["MAGNUM_LOG"] = "quiet"


def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


def _generate_fn(args):
    (
        scene,
        split,
        num_episodes_per_scene,
        skip_existing,
        is_gen_shortest_path,
        force_recreate_navmesh,
        episode_data_path,
        scene_files_path,
        config_file,
        island_sampling_type,
        scene_dataset_config_file,
        scene_dataset,
        simulator_type,
        skip_navmesh_load,
    ) = args

    scene_key = scene.split("/")[-1].split(".")[0]
    out_file = f"{episode_data_path}/{split}/content/{scene_key}.json.gz"

    # Skip scene generation if the file already exists
    if skip_existing and osp.exists(out_file):
        return

    register_plugins()

    cfg = habitat.get_config(config_file)
    with habitat.config.read_write(cfg):
        if scene_dataset_config_file:
            cfg.habitat.simulator.scene_dataset = scene_dataset_config_file
        cfg.habitat.simulator.scene = scene
        agent_config = get_agent_config(cfg.habitat.simulator)
        cfg.habitat.simulator.create_renderer = True

    sim = habitat.sims.make_sim(simulator_type, config=cfg.habitat.simulator)
    navmesh_path = scene.replace(".glb", ".navmesh")
    if not skip_navmesh_load:
        if not force_recreate_navmesh and osp.exists(navmesh_path):
            sim.pathfinder.load_nav_mesh(navmesh_path)
            print(f"Loaded navmesh from {navmesh_path}")
        else:
            print(
                f"Requested navmesh to load from {navmesh_path}. Either not found or force recreation needed. Recomputing from configured values and caching."
            )
            print("Using the following radius, height, max_climb")
            print(
                agent_config.radius,
                agent_config.height,
                agent_config.max_climb,
            )

            navmesh_settings = NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_radius = agent_config.radius
            navmesh_settings.agent_height = agent_config.height
            navmesh_settings.agent_max_climb = agent_config.max_climb
            navmesh_settings.cell_height = 0.1
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
            os.makedirs(osp.dirname(navmesh_path), exist_ok=True)
            # sim.pathfinder.save_nav_mesh(navmesh_path)

    dset = habitat.datasets.make_dataset("PointNav-v1")

    if is_gen_shortest_path:
        dset.episodes, shortest_path_lengths = list(
            zip(
                *list(
                    generate_pointnav_episode(
                        sim,
                        num_episodes_per_scene,
                        is_gen_shortest_path=is_gen_shortest_path,
                        island_sampling_type=island_sampling_type,
                    )
                )
            )
        )
        dset.episodes = list(dset.episodes)
        print(
            f"Shortest path lengths for {scene}: Min {np.min(shortest_path_lengths)}, Max {np.max(shortest_path_lengths)}, Avg {np.mean(shortest_path_lengths)}"
        )
    else:
        dset.episodes = list(
            generate_pointnav_episode(
                sim,
                num_episodes_per_scene,
                is_gen_shortest_path=is_gen_shortest_path,
                island_sampling_type=island_sampling_type,
            )
        )
    sim.close()
    if scene_dataset == "hssd":
        for ep in dset.episodes:
            scene_id = osp.join("scenes", f"{scene_key}.scene_instance.json")
            assert osp.exists(
                scene_dataset_config_file
            ), f"Scene dataset config file {scene_dataset_config_file} does not exist"
            ep.scene_id = scene_id
            ep.scene_dataset_config = scene_dataset_config_file

    else:
        for ep in dset.episodes:
            ep.scene_id = ep.scene_id[len(f"{scene_files_path}/") :]
            if scene_dataset_config_file:
                ep.scene_dataset_config = scene_dataset_config_file

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())


def generate_dataset(args):
    if args.scene_dataset != "custom":
        train_episodes_per_scene = 1e4
        eval_episodes_per_scene = 25
    else:
        train_episodes_per_scene = 1000
        eval_episodes_per_scene = 100
    if args.scene_dataset == "hm3d":
        train_scenes = glob.glob(
            f"{args.scene_files_path}/train/*/*.basis.glb"
        )
        val_scenes = glob.glob(f"{args.scene_files_path}/val/*/*.basis.glb")
        simulator_type = "CustomSim-v0"
        args.island_sampling_type = "none"
    elif args.scene_dataset == "hssd":
        scenes = glob.glob(f"{args.scene_files_path}/*.glb")
        args.island_sampling_type = "random"
        indices = np.arange(len(scenes))
        np.random.shuffle(indices)
        split_index = int(len(scenes) * 0.8)
        train_scenes = [scenes[i] for i in indices[:split_index]]
        val_scenes = [scenes[i] for i in indices[split_index:]]
        simulator_type = "CustomSim-v0"

    else:  # custom capture
        scenes = glob.glob(f"{args.scene_files_path}/*.glb")
        args.island_sampling_type = "largest"
        train_scenes = scenes
        val_scenes = scenes
        simulator_type = "CustomSim-v0"

    for split, num_episodes_per_scene, scenes in zip(
        ["train", "val"],
        [train_episodes_per_scene, eval_episodes_per_scene],
        [train_scenes, val_scenes],
    ):

        print(f"Total number of {split} scenes: {len(scenes)}")
        os.makedirs(f"{args.episode_data_path}/{split}", exist_ok=True)

        if args.use_multiprocessing:
            with multiprocessing.Pool(8) as pool, tqdm.tqdm(
                total=len(scenes)
            ) as pbar:
                args_list = [
                    (
                        scene,
                        split,
                        num_episodes_per_scene,
                        args.skip_existing,
                        args.is_gen_shortest_path,
                        args.force_recreate_navmesh,
                        args.episode_data_path,
                        args.scene_files_path,
                        args.config_file,
                        args.island_sampling_type,
                        args.scene_dataset_config_file,
                        args.scene_dataset,
                        simulator_type,
                        args.skip_navmesh_load,
                    )
                    for scene in scenes
                ]
                for _ in pool.imap_unordered(_generate_fn, args_list):
                    pbar.update()
        else:
            for scene in tqdm.tqdm(scenes):
                _generate_fn(
                    (
                        scene,
                        split,
                        num_episodes_per_scene,
                        args.skip_existing,
                        args.is_gen_shortest_path,
                        args.force_recreate_navmesh,
                        args.episode_data_path,
                        args.scene_files_path,
                        args.config_file,
                        args.island_sampling_type,
                        args.scene_dataset_config_file,
                        args.scene_dataset,
                        simulator_type,
                        args.skip_navmesh_load,
                    )
                )

        path = f"{args.episode_data_path}/{split}/{split}.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(dict(episodes=[]), f)


if __name__ == "__main__":
    # Default constants
    DEFAULT_CONFIG_FILE = "config/experiments/ddppo_pointnav.yaml"

    parser = argparse.ArgumentParser(
        description="Generate PointNav dataset splits with various options."
    )
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Enable multiprocessing for scene processing.",
    )
    parser.add_argument(
        "--is_gen_shortest_path",
        action="store_true",
        help="Generate shortest paths for episodes.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scene files which have existing episodes.",
    )
    parser.add_argument(
        "--scene_dataset",
        default="custom",
        choices=["custom", "hm3d", "hssd"],
        help="Scene dataset to use.",
    )
    parser.add_argument(
        "--force_recreate_navmesh",
        action="store_true",
        help="Whether to forcefully recreate the navmesh.",
    )
    parser.add_argument(
        "--episode_data_path",
        type=str,
        required=True,
        help="Path to save episode data.",
    )
    parser.add_argument(
        "--scene_files_path",
        type=str,
        required=True,
        help="Path to scene files.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_CONFIG_FILE,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--scene_dataset_config_file",
        type=str,
        help="Path to the scene dataset configuration file.",
    )
    parser.add_argument(
        "--skip_navmesh_load",
        action="store_true",
        help="Whether to skip navmesh loading entirely and rely on simulator navmeshes.",
    )

    args = parser.parse_args()
    generate_dataset(args)
