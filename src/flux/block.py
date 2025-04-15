import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
from .lora_controller import enable_lora, specify_lora


def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    condition_names: List[str],
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
    use_cache: bool = False,
) -> torch.FloatTensor:
    hh_l = hidden_states.shape[1]
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )

    with enable_lora(
        (attn.to_q, attn.to_k, attn.to_v), model_config.get("latent_lora", False)
    ):
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    if condition_latents is not None:
        cond_query, cond_key, cond_value = [], [], []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((attn.to_q, attn.to_k, attn.to_v), adapter_name):
                cond_query.append(attn.to_q(condition_latents[i]))
                cond_key.append(attn.to_k(condition_latents[i]))
                cond_value.append(attn.to_v(condition_latents[i]))

        cond_query = [
            each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            for each in cond_query
        ]
        cond_key = [
            each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            for each in cond_key
        ]
        cond_value = [
            each.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            for each in cond_value
        ]
        if attn.norm_q is not None:
            cond_query = [attn.norm_q(each) for each in cond_query]
        if attn.norm_k is not None:
            cond_key = [attn.norm_k(each) for each in cond_key]

    if cond_rotary_emb is not None:
        cond_query = [
            apply_rotary_emb(query, rotary_emb)
            for query, rotary_emb in zip(cond_query, cond_rotary_emb)
        ]
        cond_key = [
            apply_rotary_emb(key, rotary_emb)
            for key, rotary_emb in zip(cond_key, cond_rotary_emb)
        ]
        if use_cache:
            attn.cached = {
                "k": cond_key,
                "v": cond_value,
            }

    if condition_latents is not None:
        q_h = torch.cat([query], dim=2)
        k_h = torch.cat([key, *cond_key], dim=2)
        v_h = torch.cat([value, *cond_value], dim=2)
    else:
        q_h, k_h, v_h = query, key, value

    if (condition_latents is None) and hasattr(attn, "cached") and use_cache:
        k_h = torch.cat([key, *attn.cached["k"]], dim=2)
        v_h = torch.cat([value, *attn.cached["v"]], dim=2)
    hidden_states = F.scaled_dot_product_attention(
        q_h, k_h, v_h, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)

    t = []
    for i, condition_latent in (
        enumerate(condition_latents) if condition_latents is not None else []
    ):
        independent = model_config.get("independent_condition", False)
        q_t = cond_query[i]
        k_t = torch.cat([key, cond_key[i]], dim=2) if not independent else cond_key[i]
        v_t = (
            torch.cat([value, cond_value[i]], dim=2)
            if not independent
            else cond_value[i]
        )
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
        )
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        out = out.to(query.dtype)
        t.append(out)
    condition_latents = t or None
    if encoder_hidden_states is not None:
        ee, hh, cc = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[
                :,
                encoder_hidden_states.shape[1] : encoder_hidden_states.shape[1] + hh_l,
            ],
            hidden_states[
                :,
                encoder_hidden_states.shape[1] + hh_l :,
            ],
        )
        encoder_hidden_states = ee
        hidden_states = hh

        with enable_lora((attn.to_out[0],), model_config.get("latent_lora", False)):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None:
            final_c_outs = []
            for i, (c, adapter_name) in enumerate(
                zip(condition_latents, condition_names)
            ):
                with specify_lora((attn.to_out[0],), adapter_name):
                    c = attn.to_out[0](c)
                    c = attn.to_out[1](c)
                final_c_outs.append(c)
            condition_latents = final_c_outs

        return (
            (hidden_states, encoder_hidden_states, condition_latents)
            if condition_latents is not None
            else (hidden_states, encoder_hidden_states)
        )
    elif condition_latents is not None:
        hh, cc = (
            hidden_states[:, :hh_l],
            hidden_states[:, hh_l:],
        )
        hidden_states = hh
        return hidden_states, condition_latents
    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    condition_latents: torch.FloatTensor,
    temb: torch.FloatTensor,
    cond_temb: torch.FloatTensor,
    condition_names: List[str] = None,
    cond_rotary_emb=None,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    use_cache: bool = False,
):
    use_cond = condition_latents is not None
    with enable_lora((self.norm1.linear,), model_config.get("latent_lora", False)):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    if use_cond:
        (
            norm_condition_latents,
            cond_gate_msa,
            cond_shift_mlp,
            cond_scale_mlp,
            cond_gate_mlp,
        ) = ([] for _ in range(5))
        for i, (each, adapter_name) in enumerate(
            zip(condition_latents, condition_names)
        ):
            with specify_lora((self.norm1.linear,), adapter_name):
                res = self.norm1(each, emb=cond_temb)
            norm_condition_latents.append(res[0])
            cond_gate_msa.append(res[1])
            cond_shift_mlp.append(res[2])
            cond_scale_mlp.append(res[3])
            cond_gate_mlp.append(res[4])

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latents if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_emb=cond_rotary_emb if use_cond else None,
        condition_names=condition_names,
        use_cache=use_cache,
    )
    attn_output, context_attn_output = result[:2]
    cond_attn_output = result[2] if use_cond else None

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        # cond_attn_output = cond_gate_msa.unsqueeze(1) * cond_attn_output
        cond_attn_output = [
            cgm.unsqueeze(1) * cao for cgm, cao in zip(cond_gate_msa, cond_attn_output)
        ]
        # condition_latents = condition_latents + cond_attn_output
        condition_latents = [
            cl + cao for cl, cao in zip(condition_latents, cond_attn_output)
        ]
        if model_config.get("add_cond_attn", False):
            hidden_states += cond_attn_output

    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )
    # 3. condition_latents
    if use_cond:
        norm_condition_latents = [self.norm2(each) for each in condition_latents]
        norm_condition_latents = [
            each * (1 + csm1[:, None]) + csm[:, None]
            for each, csm1, csm in zip(
                norm_condition_latents, cond_scale_mlp, cond_shift_mlp
            )
        ]

    # Feed-forward.
    with enable_lora((self.ff.net[2],), model_config.get("latent_lora", False)):
        # 1. hidden_states
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
    # 3. condition_latents
    if use_cond:
        cond_ff_output = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.ff.net[2],), adapter_name):
                res = self.ff(norm_condition_latents[i])
                res = cond_gate_mlp[i].unsqueeze(1) * res
            cond_ff_output.append(res)

    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output
    if use_cond:
        condition_latents = [
            cl + cfo for cl, cfo in zip(condition_latents, cond_ff_output)
        ]

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    condition_names: List[str] = None,
    image_rotary_emb=None,
    condition_latents: torch.FloatTensor = None,
    cond_temb: torch.FloatTensor = None,
    cond_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
    use_cache: bool = False,
):

    using_cond = condition_latents is not None
    residual = hidden_states
    with enable_lora(
        (
            self.norm.linear,
            self.proj_mlp,
        ),
        model_config.get("latent_lora", False),
    ):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    if using_cond:
        residual_cond = condition_latents
        norm_condition_latents, cond_gate = [], []
        for each, adapter_name in zip(condition_latents, condition_names):
            with specify_lora((self.norm.linear,), adapter_name):
                res = self.norm(each, emb=cond_temb)
            norm_condition_latents.append(res[0])
            cond_gate.append(res[1])
        mlp_cond_hidden_states = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.proj_mlp,), adapter_name):
                res = self.act_mlp(self.proj_mlp(norm_condition_latents[i]))
            mlp_cond_hidden_states.append(res)

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        condition_names=condition_names,
        use_cache=use_cache,
        **(
            {
                "condition_latents": norm_condition_latents,
                "cond_rotary_emb": cond_rotary_emb if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    if using_cond:
        attn_output, cond_attn_output = attn_output

    with enable_lora((self.proj_out,), model_config.get("latent_lora", False)):
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
    if using_cond:
        condition_latents = [
            torch.cat([cao, mlp], dim=2)
            for cao, mlp in zip(cond_attn_output, mlp_cond_hidden_states)
        ]
        cond_gate = [each.unsqueeze(1) for each in cond_gate]
        t = []
        for i, adapter_name in enumerate(condition_names):
            with specify_lora((self.proj_out,), adapter_name):
                res = cond_gate[i] * self.proj_out(condition_latents[i])
                t.append(res)
        condition_latents = t
        condition_latents = [
            rc + cl for rc, cl in zip(residual_cond, condition_latents)
        ]

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states if not using_cond else (hidden_states, condition_latents)
