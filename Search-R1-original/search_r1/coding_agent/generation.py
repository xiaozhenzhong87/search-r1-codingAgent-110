"""
Coding Agent Generation Manager.
Adapts the multi-step LLM agent loop for code execution tasks using Docker sandboxes.
"""

import torch
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
from search_r1.llm_agent.tensor_helper import TensorHelper, TensorConfig
from search_r1.coding_agent.docker_sandbox import SandboxPool, ExecResult
from verl import DataProto


class CodingAgentGenerationManager(LLMGenerationManager):
    """Generation manager for coding agent that uses Docker sandboxes."""

    def __init__(self, tokenizer, actor_rollout_wg, config: GenerationConfig,
                 is_validation: bool = False):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        self.sandbox_pool = SandboxPool()
        self._batch_metadata = None
        self._current_sandbox_ids = None
        self.test_scores = None

    def set_batch_metadata(self, non_tensor_batch):
        self._batch_metadata = non_tensor_batch

    def _parse_metadata(self, idx):
        if self._batch_metadata is None:
            return {}
        metadata_raw = self._batch_metadata.get('metadata', None)
        if metadata_raw is not None:
            if isinstance(metadata_raw, np.ndarray):
                metadata_str = metadata_raw[idx]
            elif isinstance(metadata_raw, list):
                metadata_str = metadata_raw[idx]
            else:
                metadata_str = metadata_raw
            if isinstance(metadata_str, str):
                try:
                    return json.loads(metadata_str)
                except json.JSONDecodeError:
                    pass
            elif isinstance(metadata_str, dict):
                return metadata_str
        return {}

    def _init_sandboxes(self, batch_size):
        context_list = []
        harness_list = []
        for i in range(batch_size):
            meta = self._parse_metadata(i)
            context_list.append(meta.get('context', {}))
            harness_list.append(meta.get('harness', {}))
        self._current_sandbox_ids = self.sandbox_pool.create_batch(
            context_files_list=context_list,
            harness_files_list=harness_list,
        )
        self._harness_list = harness_list

    def _destroy_sandboxes(self):
        if self._current_sandbox_ids:
            self.sandbox_pool.destroy_batch(self._current_sandbox_ids)
            self._current_sandbox_ids = None

    def _postprocess_responses(self, responses):
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        processed = []
        for resp in responses_str:
            if '</bash>' in resp:
                processed.append(resp.split('</bash>')[0] + '</bash>')
            elif '<done>' in resp:
                if '</done>' in resp:
                    processed.append(resp.split('</done>')[0] + '</done>')
                else:
                    processed.append(resp.split('<done>')[0] + '<done></done>')
            else:
                processed.append(resp)
        responses_ids = self._batch_tokenize(processed)
        return responses_ids, processed

    def postprocess_predictions(self, predictions):
        actions = []
        contents = []
        for prediction in predictions:
            bash_match = re.search(r'<bash>(.*?)</bash>', prediction, re.DOTALL)
            done_match = re.search(r'<done>(.*?)</done>', prediction, re.DOTALL)
            if bash_match:
                actions.append('bash')
                contents.append(bash_match.group(1).strip())
            elif done_match:
                actions.append('done')
                contents.append(done_match.group(1).strip())
            else:
                actions.append(None)
                contents.append('')
        return actions, contents

    def execute_predictions(self, predictions, pad_token, active_mask=None, do_search=True):
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            if action == 'done':
                next_obs.append('')
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
            elif action == 'bash':
                cmd = contents[i]
                sandbox_id = self._current_sandbox_ids[i]
                sandbox = self.sandbox_pool.get_sandbox(sandbox_id)
                if sandbox:
                    result = sandbox.exec(cmd)
                    output = result.stdout
                    if result.stderr:
                        output += ("\n[stderr]: " + result.stderr) if output else result.stderr
                    if not output.strip():
                        output = "[exit code: " + str(result.exit_code) + "]"
                    next_obs.append('\n\n<observation>' + output.strip() + '</observation>\n\n')
                else:
                    next_obs.append('\n\n<observation>Error: sandbox not available</observation>\n\n')
                dones.append(0)
                valid_action.append(1)
                is_search.append(1)
            else:
                next_obs.append(
                    '\nMy previous action is invalid. '
                    'I should use <bash>command</bash> to execute a command, '
                    'or <done>summary</done> when finished. Let me try again.\n'
                )
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)

        return next_obs, dones, valid_action, is_search

    def _run_tests_for_batch(self, batch_size):
        scores = []
        for i in range(batch_size):
            if self._current_sandbox_ids and i < len(self._current_sandbox_ids):
                sandbox_id = self._current_sandbox_ids[i]
                sandbox = self.sandbox_pool.get_sandbox(sandbox_id)
                if sandbox and i < len(self._harness_list) and self._harness_list[i]:
                    try:
                        pass_rate, raw_output = sandbox.run_tests(self._harness_list[i])
                        scores.append(pass_rate)
                        continue
                    except Exception as e:
                        print(f"[WARNING] Test failed for sample {i}: {e}")
            scores.append(0.0)
        return scores

    def run_llm_loop(self, gen_batch, initial_input_ids):
        batch_size = gen_batch.batch['input_ids'].shape[0]
        self._init_sandboxes(batch_size)

        try:
            original_left_side = {
                'input_ids': initial_input_ids[:, -self.config.max_start_length:]
            }
            original_right_side = {
                'responses': initial_input_ids[:, []],
                'responses_with_info_mask': initial_input_ids[:, []]
            }

            active_mask = torch.ones(batch_size, dtype=torch.bool)
            turns_stats = torch.ones(batch_size, dtype=torch.int)
            valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
            valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
            active_num_list = [active_mask.sum().item()]
            rollings = gen_batch

            for step in range(self.config.max_turns):
                if not active_mask.sum():
                    break
                rollings.batch = self.tensor_fn.cut_to_effective_len(
                    rollings.batch,
                    keys=['input_ids', 'attention_mask', 'position_ids']
                )
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })
                gen_output = self._generate_with_gpu_padding(rollings_active)
                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(
                    gen_output.batch['responses']
                )
                responses_ids, responses_str = self.tensor_fn._example_level_pad(
                    responses_ids, responses_str, active_mask
                )
                next_obs, step_dones, step_valid, step_search = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask
                )
                curr_active_mask = torch.tensor(
                    [not d for d in step_dones], dtype=torch.bool
                )
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())
                turns_stats[curr_active_mask] += 1
                valid_action_stats += torch.tensor(step_valid, dtype=torch.int)
                valid_search_stats += torch.tensor(step_search, dtype=torch.int)
                next_obs_ids = self._process_next_obs(next_obs)
                rollings = self._update_rolling_state(
                    rollings, responses_ids, next_obs_ids
                )
                original_right_side = self._update_right_side(
                    original_right_side, responses_ids, next_obs_ids
                )

            if active_mask.sum():
                rollings.batch = self.tensor_fn.cut_to_effective_len(
                    rollings.batch,
                    keys=['input_ids', 'attention_mask', 'position_ids']
                )
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })
                gen_output = self._generate_with_gpu_padding(rollings_active)
                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(
                    gen_output.batch['responses']
                )
                responses_ids, responses_str = self.tensor_fn._example_level_pad(
                    responses_ids, responses_str, active_mask
                )
                _, step_dones, step_valid, step_search = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask,
                    do_search=False
                )
                curr_active_mask = torch.tensor(
                    [not d for d in step_dones], dtype=torch.bool
                )
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())
                valid_action_stats += torch.tensor(step_valid, dtype=torch.int)
                valid_search_stats += torch.tensor(step_search, dtype=torch.int)
                original_right_side = self._update_right_side(
                    original_right_side, responses_ids
                )

            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['valid_search_stats'] = valid_search_stats.tolist()
            print("ACTIVE_TRAJ_NUM:", active_num_list)

            print("[CodingAgent] Running test harness...")
            self.test_scores = self._run_tests_for_batch(batch_size)
            meta_info['test_scores'] = self.test_scores
            avg_score = sum(self.test_scores) / max(len(self.test_scores), 1)
            nz = sum(1 for s in self.test_scores if s > 0)
            print(f"[CodingAgent] Test scores: mean={avg_score:.3f}, nonzero={nz}/{len(self.test_scores)}")

            return self._compose_final_output(
                original_left_side, original_right_side, meta_info
            )
        finally:
            self._destroy_sandboxes()
