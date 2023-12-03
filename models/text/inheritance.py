import torch
import os
from torch import nn
import torch.nn.functional as F
from transformers import RobertaTokenizer,RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaLayer, RobertaOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,args):
        super().__init__()

        self.fc2 = nn.Linear(input_dim, output_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(output_dim)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.bn = args.if_bn
        self.dropout1 = nn.Dropout(0.2)

        self.apply(init_weights)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        if self.bn:
            x = F.gelu(self.bn1(self.fc2(x)))
        else:
            x = F.gelu(self.fc2(x))
        x = self.dropout1(x)
        return x




class Adapter(nn.Module):
    def __init__(self, D_features, act_layer=nn.GELU, skip_connect=True, in_dim =64, out_dim =64):
        super().__init__()
        self.skip_connect = skip_connect
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, in_dim)
        self.Update_mlp = nn.Linear(in_dim, out_dim)
        self.D_fc2 = nn.Linear(out_dim, D_features)
        self.dropout1 = nn.Dropout(0.2)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = F.gelu(xs)
        xs = self.dropout1(xs)
        xs = self.Update_mlp(xs)
        xs = F.gelu(xs)
        xs = self.dropout1(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class new_Layer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.Adapter = Adapter(D_features=768, skip_connect=False)
        self.output = newRobertaOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask = None,
            head_mask = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            past_key_value = None,
            output_attentions = False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]


        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )


            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]


            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value


        ada_outputs = self.Adapter(attention_output)
        layer_output = self.feed_forward_chunk(attention_output=attention_output, ada_outputs = ada_outputs)
        outputs = (layer_output,) + outputs


        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    def feed_forward_chunk(self, attention_output, ada_outputs):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, ada_outputs)
        return layer_output

class new_Layer_s(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.Adapter = Adapter(D_features=768, skip_connect=False)
        self.output = newRobertaOutput_s(config)

    def forward(
            self,
            hidden_states,
            attention_mask = None,
            head_mask = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            past_key_value = None,
            output_attentions = False,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]


        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )


            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]


            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value


        ada_outputs = None
        layer_output = self.feed_forward_chunk(attention_output=attention_output, ada_outputs = ada_outputs)
        outputs = (layer_output,) + outputs


        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs
    def feed_forward_chunk(self, attention_output, ada_outputs):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, ada_outputs)
        return layer_output

class newRobertaOutput_s(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.Adapter = Adapter(D_features=768, skip_connect=False)

    def forward(self, hidden_states, input_tensor, ada_outputs):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.Adapter(hidden_states)
        return hidden_states

class newRobertaOutput(RobertaOutput):
    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, ada_outputs):
        hidden_states = self.dense(hidden_states)
        hidden_states = hidden_states + ada_outputs
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Text_Model1(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=True)

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            past_key_values = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = True,
    ):
        input_ids = input_ids.long()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)


        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)



        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class RobertaEncoder1(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([new_Layer_s(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        head_mask = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        past_key_values = None,
        use_cache = None,
        output_attentions = False,
        output_hidden_states = False,
        return_dict = True,
    ) :
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states


class Text_Model2(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=True)

    def forward(
            self,
            h_s,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False


        if h_s is not None:
            input_shape = h_s.size()
            self.warn_if_padding_and_no_attention_mask(h_s, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length, dim= input_shape
        device = h_s.device if h_s is not None else inputs_embeds.device


        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)


        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)


        encoder_outputs = self.encoder(
            h_s,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
class RobertaEncoder2(RobertaEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer[-1] = new_Layer(config)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


def t_model_p1(args):
    config1 = RobertaConfig.from_json_file('')
    config1.num_hidden_layers = 1
    model1 = Text_Model1(config1)
    model1.encoder = RobertaEncoder1(config1)
    del config1

    model1.requires_grad_(False)
    for i, block in enumerate(model1.encoder.layer):
        block.Adapter.requires_grad_(True)

    non_trainable_params = sum(p.numel() for p in model1.parameters() if not p.requires_grad)
    print("text:", non_trainable_params)
    return model1
def t_model_p2(args):
    client_part2_t = Classifier(input_dim=args.t_fea_dim, hidden_dim=args.t_hid_dim, output_dim=args.t_class_num, args=args)

    client_part2_t.apply(init_weights)
    client_part2_t.requires_grad_(True)
    return client_part2_t
def t_model_p3(args):
    config2 = RobertaConfig.from_json_file('')
    config2.num_hidden_layers = 12
    model2 = Text_Model2(config2)
    model2.encoder = RobertaEncoder2(config2)
    del model2.encoder.layer[0]

    model2.requires_grad_(False)
    model2.encoder.layer[-1].Adapter.requires_grad_(True)

    del config2
    return model2


