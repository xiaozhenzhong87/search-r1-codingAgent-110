#!/usr/bin/env python3
"""
训练完成后,从resume的旧run中提取step 201+的数据,创建独立的新WandB run
"""

import wandb
import pandas as pd
import sys

def create_separate_run_from_step201():
    """
    从旧run中提取step 201+的数据,创建新的独立run
    """
    
    print("=" * 70)
    print("  从旧run中分离step 201+的数据创建新run")
    print("=" * 70)
    print()
    
    # 1. 连接到API
    print("[1/4] 连接到WandB API...")
    api = wandb.Api()
    
    # 2. 获取旧run
    print("[2/4] 获取旧run数据...")
    old_run_path = "xiaozhenzhong87-uestc/Search-R1/b6q2bvxz"
    
    try:
        old_run = api.run(old_run_path)
        print(f"✓ 找到旧run: {old_run.name}")
        print(f"  URL: {old_run.url}")
    except Exception as e:
        print(f"✗ 无法获取旧run: {e}")
        return False
    
    # 3. 提取历史数据
    print("[3/4] 提取历史数据...")
    history = old_run.history()
    print(f"  总共 {len(history)} 条记录")
    
    # 过滤step >= 201的数据
    if '_step' in history.columns:
        filtered_data = history[history['_step'] >= 201].copy()
        print(f"  提取 step >= 201 的数据: {len(filtered_data)} 条")
    else:
        print("✗ 未找到_step字段,无法过滤")
        return False
    
    if len(filtered_data) == 0:
        print("✗ 没有找到step >= 201的数据")
        return False
    
    # 4. 创建新run
    print("[4/4] 创建新的WandB run...")
    new_run = wandb.init(
        project="Search-R1",
        name="nq-ppo-from-step200-improved-reward",
        config={
            "model": "Qwen2.5-7B-Instruct",
            "base_model": "/ssd1/zz/models/Qwen/Qwen2.5-7B-Instruct",
            "kl_type": "low_var_kl",
            "kl_coef": 0.001,
            "resumed_from_checkpoint": "global_step_200",
            "reward_version": "v2_penalty",
            "format_score": 0.0,
            "penalty_ratio": 0.2,
            "original_run": old_run_path,
            "data_source": f"Extracted from {old_run.name} (step >= 201)",
        },
        resume=False  # 不要resume
    )
    
    print(f"✓ 新run已创建: {new_run.name}")
    print(f"  URL: {new_run.url}")
    print()
    
    # 5. 上传数据
    print("上传数据到新run...")
    
    # 重新调整step (从201开始 -> 从0开始,或者保持201开始)
    # 选项A: 保持原始step (201, 202, 203...)
    use_original_step = True
    
    # 选项B: 重新从0开始 (0, 1, 2...)
    # use_original_step = False
    
    for idx, row in filtered_data.iterrows():
        # 提取metrics (排除WandB内部字段)
        metrics = {}
        for col in filtered_data.columns:
            if not col.startswith('_') and pd.notna(row[col]):
                try:
                    metrics[col] = float(row[col])
                except (ValueError, TypeError):
                    pass  # 跳过非数值字段
        
        # 确定step
        if use_original_step:
            step = int(row['_step'])
        else:
            step = int(row['_step']) - 201  # 从0开始
        
        # 上传
        if metrics:
            wandb.log(metrics, step=step)
        
        # 显示进度
        if (idx % 100 == 0):
            print(f"  已上传 {idx}/{len(filtered_data)} 条记录...")
    
    print(f"✓ 完成! 共上传 {len(filtered_data)} 条记录")
    print()
    
    # 6. 完成
    new_run.finish()
    
    print("=" * 70)
    print("  新run创建成功!")
    print("=" * 70)
    print()
    print(f"新run URL: {new_run.url}")
    print()
    print("现在您可以在WandB中对比两条曲线:")
    print(f"  - 原训练: {old_run.name} (step 0-500+)")
    print(f"  - 新训练: {new_run.name} (step 201+ 或 0+)")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = create_separate_run_from_step201()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
