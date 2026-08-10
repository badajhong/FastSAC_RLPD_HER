from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="ldt29@Tsinghua,Dongting Li",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb; platform_machine != 'aarch64'",
        "wandb==0.19.11; platform_machine == 'aarch64'",
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
        "onnx==1.18.0; platform_machine == 'aarch64'",
        "onnxscript==0.6.2",
        "onnxruntime==1.24.4",
        "torch==2.7.0; platform_machine != 'aarch64'",
        "torch>=2.9.0,<2.10; platform_machine == 'aarch64'",
        "torchrl==0.7.0; platform_machine != 'aarch64'",
        "torchrl==0.8.0; platform_machine == 'aarch64'",
        "tensordict==0.7.0; platform_machine != 'aarch64'",
        "tensordict==0.8.1; platform_machine == 'aarch64'",
    ],
)
