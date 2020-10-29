#!/usr/bin/env python

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from torch.utils.hipify import hipify_python

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 5], "Requires PyTorch >= 1.5"


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "votenet", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("VN_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "votenet", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    is_rocm_pytorch = False
    if torch_ver >= [1, 5]:
        from torch.utils.cpp_extension import ROCM_HOME

        is_rocm_pytorch = (
            True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
        )

    if is_rocm_pytorch:
        hipify_python.hipify(
            project_directory=this_dir,
            output_directory=this_dir,
            includes="/votenet/layers/csrc/*",
            show_detailed=True,
            is_pytorch_extension=True,
        )

        # Current version of hipify function in pytorch creates an intermediate directory
        # named "hip" at the same level of the path hierarchy if a "cuda" directory exists,
        # or modifying the hierarchy, if it doesn't. Once pytorch supports
        # "same directory" hipification (https://github.com/pytorch/pytorch/pull/40523),
        # the source_cuda will be set similarly in both cuda and hip paths, and the explicit
        # header file copy (below) will not be needed.
        source_cuda = glob.glob(path.join(extensions_dir, "**", "hip", "*.hip")) + glob.glob(
            path.join(extensions_dir, "hip", "*.hip")
        )

        shutil.copy(
            "votenet/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h",
            "votenet/layers/csrc/box_iou_rotated/hip/box_iou_rotated_utils.h",
        )

    else:
        source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
            path.join(extensions_dir, "*.cu")
        )

    sources = [main_source] + sources
    sources = [
        s
        for s in sources
        if not is_rocm_pytorch or torch_ver < [1, 7] or not s.endswith("hip/vision.cpp")
    ]

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "votenet._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="votenet",
    version=get_version(),
    author="Ming Yang (ymviv@qq.com)",
    url="https://github.com/vivym/votenet",
    description="VoteNet",
    packages=find_packages(exclude=("configs", "tests*")),
    python_requires=">=3.6",
    install_requires=[
        # Do not add opencv here. Just like pytorch, user should install
        # opencv themselves, preferrably by OS's package manager, or by
        # choosing the proper pypi package name at https://github.com/skvark/opencv-python
        "termcolor>=1.1",
        "Pillow-SIMD>=7.0",  # or use pillow-simd for better performance
        "yacs>=0.1.6",
        "tabulate",
        "cloudpickle",
        "matplotlib",
        "mock",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore>=0.1.1",
    ],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
