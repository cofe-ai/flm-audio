# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Callable
import torch
import time
from transformers.cache_utils import DynamicCache
from ..third_party.moshi.utils.sampling import sample_token
from ..third_party.moshi.modules.streaming import StreamingModule
from ..utils import log

@dataclass
class _LMGenState:
    cache: torch.Tensor
    initial: torch.Tensor
    forward_text: Callable
    depformer_step: Callable
    past_key_values: DynamicCache
    offset: int
    text_tokens_cache: List
    audio_speak_tokens_cache: List
    def reset(self):
        self.offset = 0
        self.past_key_values = DynamicCache()
        self.audio_speak_tokens_cache = []
        self.text_tokens_cache = []


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model,
        text_tokenizer,
        use_sampling: bool = True,
        temp_audio: float = 0.5,
        temp_text: float = 0.01,
        top_k_audio: int = 1,
        top_k_text: int = 1,
        top_p_audio: float = 0.9,
        top_p_text: float = 0.9,
        repetition_penalty_audio: float = 1.2,
        repetition_penalty_text: float = 1.03,
        repetition_penalty_window_audio: int = 50,
        repetition_penalty_window_text: int = 50,
    ):
        super().__init__()

        self.lm_model = lm_model
        self.text_tokenizer = text_tokenizer

        self.use_sampling = use_sampling
        self.temp_audio = temp_audio
        self.temp_text = temp_text
        self.top_k_audio = top_k_audio
        self.top_k_text = top_k_text
        self.top_p_audio = top_p_audio
        self.top_p_text = top_p_text
        self.repetition_penalty_audio = repetition_penalty_audio
        self.repetition_penalty_text = repetition_penalty_text
        self.repetition_penalty_window_audio = repetition_penalty_window_audio
        self.repetition_penalty_window_text = repetition_penalty_window_text
        self.delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        self.max_delay = max(
            self.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            self.delays, device=lm_model.device, dtype=torch.long
        )

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        log("info", f"in _init_streaming_state")
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, 17, self.max_delay + 20),
            0,
            device=lm_model.device,
            dtype=torch.long,
        )
        past_key_values = DynamicCache()

        lm_model.forward = torch.compile(lm_model.forward)
        lm_model.forward_audio = torch.compile(lm_model.forward_audio)

        log("info", f"torch.compile begin")
        t_s = time.time()
        # self.step = torch.compile(self.step)
        log("info", f"torch.compile done: {time.time() - t_s:.1f}s")

        return _LMGenState(cache, initial, lm_model.forward, self.depformer_step, past_key_values, offset=0, audio_speak_tokens_cache=[], text_tokens_cache=[])

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        assert S == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.config.aud_channel
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        CT = state.cache.shape[2]

        for q_other in range(input_tokens.shape[1]):
            k = 1 + lm_model.config.aud_channel + q_other
            delay = self.delays[k]
            write_position = (state.offset + delay) % CT
            state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]

        position = state.offset % CT
        for k, delay in enumerate(self.delays):
            if state.offset == 0 and k <= 8 or state.offset < delay and k > 8:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position : position + 1]
        input_ = input_.permute(0, 2, 1).contiguous()
        model_input = {
            'input_ids': input_[:, :, 0],
            'speak_ids': input_[:, :, 1:9],
            'listen_ids': input_[:, :, 9:],
            'past_key_values': state.past_key_values,
            'use_cache': True,
        }
        if False and state.past_key_values is not None and len(state.past_key_values) > 0:
            log("info", f"kv cahce: [{len(state.past_key_values)},{len(state.past_key_values[0])}] {state.past_key_values[0][0].shape}")

        t_s = time.time()
        lm_outputs = state.forward_text(**model_input)
        # log("info", f"lm.forward_text: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()
        state.past_key_values = lm_outputs.past_key_values
        last_text_token_logits = lm_outputs.logits[:, -1:, :]
        last_hidden_states = lm_outputs.hidden_states[:, -1, :]
        if len(state.text_tokens_cache) == 0:
            rep_window_input_ids = None
        else:
            rep_window_input_ids = torch.concat(state.text_tokens_cache, dim=0).to(dtype=torch.long, device=last_hidden_states.device).unsqueeze(dim=0)
        sampled_text_token = sample_token(
            last_text_token_logits.squeeze(1),
            use_sampling=self.use_sampling,
            temp=self.temp_text,
            top_p=self.top_p_text,
            top_k=self.top_k_text,
            repetition_penalty=self.repetition_penalty_text if rep_window_input_ids is not None else 1.0,
            input_ids=rep_window_input_ids,
        )
        # log("info", f"sample_text: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()

        state.text_tokens_cache.append(sampled_text_token)
        if len(state.text_tokens_cache) > self.repetition_penalty_window_text:
            state.text_tokens_cache.pop(0)

        sampled_text_token = sampled_text_token[0]  # shape is [B]

        audio_tokens = state.depformer_step(last_hidden_states)
        # log("info", f"depformer_step: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()

        state.offset += 1
        position = state.offset % CT

        state.cache[:, 0, position] = sampled_text_token
        state.cache[:, 1:9, position] = audio_tokens

        if state.offset <= self.max_delay:
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: 8+1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)

        return out

    def depformer_step(
        self,
        last_hidden_states: torch.Tensor,
    ):
        state = self._streaming_state
        decoded_audio_tokens = torch.empty((1, 0), dtype=torch.long, device=last_hidden_states.device) # [B, 0]

        if len(state.audio_speak_tokens_cache) == 0:
            rep_window_input_ids = None
        else:
            rep_window_input_ids = torch.concat(state.audio_speak_tokens_cache, dim=0).unsqueeze(0).permute(0, 2, 1).to(dtype=torch.long, device=last_hidden_states.device)
        for i in range(self.lm_model.config.aud_channel):
            t_s = time.time()
            audio_logits = self.lm_model.forward_audio(
                transformer_output_states=last_hidden_states,
                audio_input_ids=decoded_audio_tokens
            ) # [B, K, vocab_size+1]: logis
            # if i == 0:
                # log("info", f"forward_audio[{i}]: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()
            current_aud_logit = audio_logits[:, -1:, :]

            sampled_audio_token = sample_token(
                current_aud_logit.squeeze(1), # [B, 1, vocab_size+1]
                use_sampling=self.use_sampling,
                temp=self.temp_audio,
                top_p=self.top_p_audio,
                top_k=self.top_k_audio,
                repetition_penalty=self.repetition_penalty_audio if i<=3 and rep_window_input_ids is not None else 1.0,
                input_ids=rep_window_input_ids[:, i] if rep_window_input_ids is not None else None
                ).unsqueeze(1)   # [B, 1, 1(k)]   ->[B, 1(k)]
            decoded_audio_tokens = torch.concat([decoded_audio_tokens, sampled_audio_token], dim=1)   # [B, 0] [B, 1(k)]
            # if i == 0:
                # log("info", f"sample_audio[{i}]: {1000 * (time.time() - t_s):.1f}ms"); t_s = time.time()

        state.audio_speak_tokens_cache.append(decoded_audio_tokens)

        if len(state.audio_speak_tokens_cache) > self.repetition_penalty_window_audio:
            state.audio_speak_tokens_cache.pop(0)
        return decoded_audio_tokens