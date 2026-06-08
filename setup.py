from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="ldt29@Tsinghua,Dongting Li",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "einops",
        "av", # for moviepy
        "pandas",
        "termcolor",
        "setproctitle",
        "pygame", # for game controller
        "mujoco",
        "xxhash",
        "onnxscript==0.6.2",
        "onnxruntime==1.24.4",
        "torch==2.7.0",
        # "torch==2.8.0",
        # "torchvision",
        "torchrl==0.7.0",
        "tensordict==0.7.0",
    ],
)
