"""Monkeypatch helpers for Qwen, LLaMA, and Mistral attention backends."""

from typing import List, Optional, Tuple, Union
import warnings

import torch
import transformers
from torch.nn import CrossEntropyLoss

from mmshift.qwen_model import (
    adakv_qwen_forward,
    qwen_flash_attn2_forward_AdaKV,
    qwen_flash_attn2_forward_keydiff,
    qwen_flash_attn2_forward_PyramidKV,
    qwen_flash_attn2_forward_ShiftKV,
    qwen_flash_attn2_forward_SnapKV,
    qwen_flash_attn2_forward_SparseMM,
    qwen_flash_attn2_forward_streamingllm,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    BaseModelOutputWithPast,
    Qwen2_5_VLCausalLMOutputWithPast,
    _flash_attention_forward,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


def forwardllm(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    """Patched Qwen2.5-VL LM forward that threads image token metadata."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        image_token_positions=image_token_positions,
        image_token_lengths=image_token_lengths,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


def forwardllm2(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """Patched Qwen2.5-VL decoder forward that propagates image token metadata."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                image_token_positions,
                image_token_lengths,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                image_token_positions=image_token_positions,
                image_token_lengths=image_token_lengths,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def forwarddec(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """Patched Qwen2.5-VL decoder layer forward with visual-span residual handling."""
    del kwargs

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        image_token_positions=image_token_positions,
        image_token_lengths=image_token_lengths,
    )

    if residual.shape != hidden_states.shape:
        start = image_token_positions[0] + 1
        end = image_token_positions[0] + image_token_lengths[0] - 1
        residual = torch.cat([residual[:, :start], residual[:, end:]], dim=1)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    image_token_positions: Optional[torch.LongTensor] = None,
    image_token_lengths: Optional[torch.LongTensor] = None,
):
    """Patched Qwen2.5-VL flash-attention forward."""
    del use_cache, image_token_positions, image_token_lengths

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

def replace_llama_attn_with_flash_attn():
    """Install the Qwen2.5-VL flash-attention monkeypatch set."""
    cuda_major, _cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523349593"
        )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = forward
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = forwardllm2
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec

def replace_llama_attn_with_flash_attnv2():
    """Install the older Qwen2-VL flash-attention monkeypatch set."""
    cuda_major, _cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523349593"
        )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = forward
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = forwardllm
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = forwardllm2
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLDecoderLayer.forward = forwarddec
def replace_llama(method):
    """Apply a LLaMA monkeypatch for a named KV-cache method.

    Args:
        method: Patch selector such as ``snapkv``, ``pyramidkv``, ``adakv``,
            ``sparsemm``, ``shiftkv``, ``mask``, or ``fullkv``.
    """
    if method == "snapkv":
        print("Using SnapKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SnapKV
    elif method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_PyramidKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")  # Dynamic decoder layer during LLaMA forward.
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward1
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparseMM
        transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = custom_LlamaDecoderLayer_forward
    elif method == "shiftkv":
        print("Using SparseMM_query!")  # Dynamic LLaMA forward.
        transformers.models.llama.modeling_llama.LlamaModel.forward = adaptive_LlamaModel_forward
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_SparseMM_query
    elif method == "mask":
        print("Mask Head")
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama_new


def replace_mistral(method):
    """Apply a Mistral monkeypatch for a named KV-cache method.

    Args:
        method: Patch selector such as ``snapkv``, ``pyramidkv``, ``adakv``,
            ``sparsemm``, ``mask``, or ``fullkv``.
    """

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SparseMM

    elif method == "mask":
        print("Mask Head")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new

def replace_qwen(method):
    """Apply a Qwen2.5-VL monkeypatch for a named KV-cache method.

    Args:
        method: Patch selector such as ``shiftkv``, ``snapkv``, ``sparsemm``,
            ``adakv``, ``keydiff``, ``streamingllm``, or ``expectedAttention``.
    """

    if method == "streamingllm":
        print("Using streamingllm")

        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = forwardllm2
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_streamingllm
    if method == "keydiff":
        print("Using keydiff!")

        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = forwardllm2
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_keydiff
    if method == "snapkv":
        print("Using SnapKV!")

        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = forwardllm2
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM! 111")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec
    elif method == "shiftkv":
        print("Using shiftkv!")  # Dynamic visual-token handling during Qwen forward.

        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = forwardllm
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = forwardllm2
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = forwarddec
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_ShiftKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = adakv_qwen_forward
