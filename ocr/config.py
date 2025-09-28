#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

# 默认配置
DEFAULT_CONFIG = {
    # 并行设备数量，0表示使用CPU
    "PARALLEL_DEVICES": int(os.getenv("PARALLEL_DEVICES", "0")),
    
    # 模型下载源
    "HF_ENDPOINT": os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
    
    # 默认模型目录
    "DEFAULT_MODEL_DIR": "models",
    
    # OCR相关参数
    "DROP_SCORE": 0.5,
    "DET_LIMIT_SIDE_LEN": 960,
    "DET_LIMIT_TYPE": "max",
    "DET_THRESH": 0.3,
    "DET_BOX_THRESH": 0.5,
    "DET_UNCLIP_RATIO": 1.5,
    
    # 识别相关参数
    "REC_IMAGE_SHAPE": [3, 48, 320],
    "REC_BATCH_NUM": 16,
}
