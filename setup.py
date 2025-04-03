# Copyright 2023 The EffOCR team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
import os 


pkg_name = 'efficient_ocr'
libinfo_py = os.path.join('src', pkg_name, '__init__.py')
libinfo_content = open(libinfo_py, 'r', encoding='utf8').readlines()
# version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
# exec(version_line)  # gives __version__

setup(name         = "efficient_ocr",
      version      = "0.0.13",
      author       = "Tom Bryan, Abhishek Arora, Jacob Carlson, Ethan C. Tan",
      author_email = "bryanptom@gmail.com",
      license      = "Apache-2.0",
      url          = "https://github.com/dell-research-harvard/efficient_ocr",
      package_dir  = {"": "src"},
      packages     = find_packages("src"),
      description  = "Efficient OCR",
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      python_requires='>=3.6',
      install_requires=[
        "numpy", 
        "opencv-python",
        "scipy",
        "pandas",
        "pillow",
        "pyyaml>=5.1",
        "iopath",
        "pdfplumber",
        "pdf2image",
        'onnxruntime',
        'onnx', 
        'faiss-cpu',
        'yolov5<=7.0.10',
        'timm',
        'huggingface_hub==0.24.7',
        'kornia',
        'pytorch_metric_learning==1.6.3',
        'transformers==4.45.0',
        'albumentations',
        'wandb',
        'requests',
      ],
      include_package_data=True
)