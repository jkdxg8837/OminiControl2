# OminiControl2 (Early Access Version)

Thank you for trying OminiControl2! This is an early access version with experimental features. Some functionality may not work perfectly yet. If you encounter any issues, please report them to help us improve the system.

## Contact

For questions or issues:
- **Email**: Yuanshi9815@outlook.com
- **WeChat**: +65 88200624

## TL;DR
OminiControl2 improves efficiency over OminiControl1 through two main approaches:
- **Feature Reusing**: Cache conditional features across timesteps to avoid redundant computation
- **Compact Token Representation**: Reduce token count through compression and pruning techniques

Use `cache_per_n_steps` parameter for immediate efficiency gains with existing models. For research purposes, explore asymmetric attention masking with `independent_condition: True`.

## Key Features

### 1. Feature Reusing
#### Conditional Feature Caching
To optimize inference performance by reusing features, you can use the `cache_per_n_steps` parameter in the `generate` function.

For example:
- Set `cache_per_n_steps=5` to recalculate conditional image features every 5 steps
- This is compatible with previous OminiControl1 models as well

For maximum efficiency, set `cache_per_n_steps` to a value larger than your total inference steps to compute conditional features only once. Note that single-computation caching works best with models trained using **asymmetric attention masking**.

#### Asymmetric Attention Masking
To train a new model with asymmetric attention masking:
- Set `independent_condition: True` in your config file
- Example configuration files are available in the `train/config/omini2` directory

> **⚠️ Quality Notice:** Please note that Asymmetric Attention Masking may currently cause quality degradation. Our ongoing research is working to address this issue. This feature is recommended for research purposes only. For production use cases, we suggest using only the Caching mechanism without asymmetric masking to achieve the best balance between efficiency and quality (i.e., continuing to use the OminiControl1 training code).

### 2. Compact Token Representation

1. **Compression with Position Correcting** (included in this early access)
   - Enable by setting a lower `condition_size` in your config file
   - Adjust the `position_scale` parameter to maintain proper spatial alignment

2. **Token Pruning** (coming soon)
   - Currently under optimization
   - Will allow selective removal of non-informative tokens

3. **Token Integration for Mask Repainting** (coming soon)
   - Specialized optimization for inpainting tasks
   - Currently being refined for optimal performance