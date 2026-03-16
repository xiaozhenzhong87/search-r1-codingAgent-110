"""
使用答错的题目进行RL训练的脚本
基于现有的train_ppo.sh，但使用答错的题目作为训练数据
"""
import os
import subprocess
import argparse


def train_rl_on_wrong(
    base_model: str,
    rl_data_file: str,
    val_data_file: str,
    experiment_name: str,
    num_gpus: int = 8,
    search_url: str = "http://127.0.0.1:8000/retrieve",
    topk: int = 3,
    total_epochs: int = 15,
    total_training_steps: int = 500,
    **kwargs
):
    """
    使用答错的题目进行RL训练
    
    这个函数会调用verl的PPO训练器，类似于train_ppo.sh的逻辑
    但使用答错的题目作为训练数据
    """
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
    env['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    env['PYTHONUNBUFFERED'] = '1'
    
    # 构建训练命令
    cmd = [
        'python3', '-m', 'verl.trainer.main_ppo',
        f'data.train_files={rl_data_file}',
        f'data.val_files={val_data_file}',
        'data.train_data_num=null',
        'data.val_data_num=null',
        'data.train_batch_size=512',
        'data.val_batch_size=256',
        'data.max_prompt_length=4096',
        'data.max_response_length=500',
        'data.max_start_length=2048',
        'data.max_obs_length=500',
        'data.shuffle_train_dataloader=True',
        'algorithm.adv_estimator=gae',
        f'actor_rollout_ref.model.path={base_model}',
        'actor_rollout_ref.actor.optim.lr=1e-6',
        'actor_rollout_ref.model.enable_gradient_checkpointing=true',
        'actor_rollout_ref.model.use_remove_padding=True',
        'actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285',
        'actor_rollout_ref.actor.ppo_mini_batch_size=256',
        'actor_rollout_ref.actor.ppo_micro_batch_size=64',
        'actor_rollout_ref.actor.fsdp_config.param_offload=true',
        'actor_rollout_ref.actor.fsdp_config.grad_offload=true',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=true',
        'actor_rollout_ref.rollout.log_prob_micro_batch_size=128',
        'actor_rollout_ref.rollout.tensor_model_parallel_size=1',
        'actor_rollout_ref.rollout.name=vllm',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.6',
        'actor_rollout_ref.ref.log_prob_micro_batch_size=128',
        'actor_rollout_ref.ref.fsdp_config.param_offload=True',
        'actor_rollout_ref.rollout.n_agent=1',
        'actor_rollout_ref.rollout.temperature=1',
        'actor_rollout_ref.rollout.top_p=1.0',
        'actor_rollout_ref.actor.state_masking=true',
        'critic.optim.lr=1e-5',
        'critic.model.use_remove_padding=True',
        'critic.optim.lr_warmup_steps_ratio=0.015',
        f'critic.model.path={base_model}',
        'critic.model.enable_gradient_checkpointing=true',
        'critic.ppo_micro_batch_size=8',
        'critic.model.fsdp_config.param_offload=true',
        'critic.model.fsdp_config.grad_offload=true',
        'critic.model.fsdp_config.optimizer_offload=true',
        'algorithm.kl_ctrl.kl_coef=0.001',
        'algorithm.no_think_rl=false',
        'trainer.critic_warmup=0',
        "trainer.logger=['console','wandb']",
        '+trainer.val_only=false',
        '+trainer.val_before_train=true',
        'trainer.default_hdfs_dir=null',
        f'trainer.n_gpus_per_node={num_gpus}',
        'trainer.nnodes=1',
        'trainer.save_freq=100',
        'trainer.test_freq=50',
        'trainer.project_name=Search-R1',
        f'trainer.experiment_name={experiment_name}',
        f'trainer.total_epochs={total_epochs}',
        f'trainer.total_training_steps={total_training_steps}',
        'trainer.default_hdfs_dir=null',
        f'trainer.default_local_dir=verl_checkpoints/{experiment_name}',
        'max_turns=4',
        f'retriever.url={search_url}',
        f'retriever.topk={topk}',
    ]
    
    # 添加额外的kwargs参数
    for key, value in kwargs.items():
        cmd.append(f'{key}={value}')
    
    print("执行RL训练命令:")
    print(' '.join(cmd))
    print()
    
    # 执行训练
    log_file = f'{experiment_name}.log'
    with open(log_file, 'w') as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    
    print(f"训练完成！日志保存到: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用答错的题目进行RL训练")
    parser.add_argument("--base_model", type=str, required=True,
                       help="基础模型路径（SFT后的模型）")
    parser.add_argument("--rl_data_file", type=str, required=True,
                       help="RL训练数据文件路径（答错的题目）")
    parser.add_argument("--val_data_file", type=str, required=True,
                       help="验证数据文件路径")
    parser.add_argument("--experiment_name", type=str, required=True,
                       help="实验名称")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="GPU数量")
    parser.add_argument("--search_url", type=str, default="http://127.0.0.1:8000/retrieve",
                       help="搜索服务URL")
    parser.add_argument("--topk", type=int, default=3,
                       help="检索topk结果")
    parser.add_argument("--total_epochs", type=int, default=15,
                       help="总训练轮数")
    parser.add_argument("--total_training_steps", type=int, default=500,
                       help="总训练步数")
    
    args = parser.parse_args()
    
    train_rl_on_wrong(
        base_model=args.base_model,
        rl_data_file=args.rl_data_file,
        val_data_file=args.val_data_file,
        experiment_name=args.experiment_name,
        num_gpus=args.num_gpus,
        search_url=args.search_url,
        topk=args.topk,
        total_epochs=args.total_epochs,
        total_training_steps=args.total_training_steps
    )
