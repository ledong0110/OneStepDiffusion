SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score
Distillation, Thuan Hoang Nguyen and Anh Tran, VinAI Research, CVPR 2024.

Reference repository: https://github.com/VinAIResearch/SwiftBrush
Pinned commit: a346fec56484a5de54e062b1b4251f1af1e0bce0

`LICENSE.md` is copied unmodified from that commit. The upstream README
identifies CC BY-NC 4.0; its LICENSE.md identifies CC BY-NC-SA 4.0. Its trainer
retains this original header (with an Apache 2.0 copy alongside this notice):

    Copyright 2023 The HuggingFace Inc. team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

`src/few_step_diffusion/sd_swiftbrush.py` is a source-recipe port with project
integration changes documented in `docs/SWIFTBRUSH_SOURCE_AUDIT.md`.
`scripts/audit_swiftbrush.py` downloads and verifies the pinned trainer into the
ignored artifacts directory and extracts its predictor/loss expressions for
parity checks. The full upstream trainer is not vendored into this repository.
