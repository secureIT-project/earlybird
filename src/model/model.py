import argparse
import copy
import logging
from typing import Tuple, List, NoReturn

import torch
import transformers
from torch import nn

logger = logging.getLogger(__name__)


# RobertaClassificationHead is taken
# from https://github.com/huggingface/transformers/blob/f0d496828d3da3bf1e3c8fbed394d7847e839fa6/src/transformers/models/roberta/modeling_roberta.py#L1435
# and is modified for any BH-dimensional input, not only CLS token
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for classification layers.
        Update parameters of the classification head based on input data (`features`)"""
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def delete_encoder_layers(model: torch.nn.Module, num_layers_to_keep: int) -> torch.nn.Module:
    """Prune layers of the model after `num_layers_to_keep`"""
    module_list_orig = model.encoder.layer
    module_list_target = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(num_layers_to_keep):
        module_list_target.append(module_list_orig[i])

    # create a copy of the model, modify it with the new list, and return
    model_target = copy.deepcopy(model)
    model_target.encoder.layer = module_list_target
    return model_target

class DropoutAndLinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step for classification layers."""
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EncoderWithLayerCombination(nn.Module):
    def __init__(self, args: argparse.Namespace):
        logging.info(f"Initializing the model with layer combination type {args.combination_type}")
        super().__init__()
        self.num_classes = args.num_classes
        self.combination_type = args.combination_type
        model_path_or_name = args.model_path

        # TODO: for codet5p see DefectModel at https://github.com/salesforce/CodeT5/blob/430d7e358e41903357fd07ab4d14ca6fbbd03b0a/CodeT5/models.py
        self.base_model = transformers.AutoModel.from_pretrained(model_path_or_name)

        if 'cutoff_layers' in self.combination_type:
            self.base_model = delete_encoder_layers(self.base_model, args.hidden_layer_to_use)

        self.output_combination = {
            'cutoff_layers_one_layer_cls': OneLayerCLS,
            'one_layer_w_sum_tokens': OneLayerWSumTokens,
            'one_layer_max_pool_tokens': OneLayerMaxPoolTokens,
            'one_layer_cls': OneLayerCLS,
            'w_sum_tokens_w_sum_layers': WSumTokensWSumLayers,
            'max_pool_layers_w_sum_tokens': MaxPoolLayersWSumTokens,
            'w_sum_layers_w_sum_tokens': WSumLayersWSumTokens,
            'max_pool_tokens_w_sum_layers': MaxPoolTokensWSumLayers,
            'max_pool_layers_max_pool_tokens': MaxPoolLayersMaxPoolTokens,
            'w_sum_cls': WSumCLS,
            'max_pool_cls': MaxPoolCLS,
            'last_layer_cls': LastLayerCLS
        }[args.combination_type](args)

        self.classifier = {
            'one_linear_layer': DropoutAndLinearClassificationHead,
            'roberta_classification_head': RobertaClassificationHead
        }[args.clf_architecture](args)

        self.freeze(args)
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self, args: argparse.Namespace) -> None:
        """Freeze base mode layers to fine-tune only the classification head."""
        logger.info('\n\nFreezing the following layers of the encoder model')
        # Freezes the full codebert model and trains only weights on top of codebert
        if args.freeze_base_model:
            for name, param in list(self.base_model.named_parameters()):
                param.requires_grad = False
                logger.info(name)
        elif args.freeze_embeddings:
            for name, param in list(self.base_model.named_parameters()):
                if name.startswith("embeddings"):
                    param.requires_grad = False
                    logger.info(name)
        logger.info('Finished freezing\n\n')

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Forward step of the model."""
        outputs = self.base_model(
            input_ids=inputs.squeeze(),
            attention_mask=masks.squeeze(),
            output_hidden_states=(self.combination_type != 'last_layer_cls'))

        # Hidden state combination
        sequence_representation = self.output_combination(outputs, masks)
        logits = self.classifier(sequence_representation)

        return logits


def zero_out_padding_and_special_tokens(
        hidden_states: Tuple[torch.FloatTensor],
        masks: torch.Tensor,
        special_tokens_to_zero: bool
) -> List[torch.Tensor]:
    """Set padding and other special tokens to zero."""
    if special_tokens_to_zero:
        for m in masks:
            m[0][0] = 0
            m[0][m.sum(dim=1).item() - 1] = 0
    revert_masks = 1 - masks.squeeze().unsqueeze(-1)
    x = [hidden_states[int(i)].masked_fill(revert_masks.to(torch.bool), 0.0) for i in range(len(hidden_states))]
    return x


class Combination(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # Learnable weights in the weighted sum over hidden layers
        # Number of layers = 12 in CodeBert
        self.w_layers = nn.Parameter(torch.rand(12))

        # Learnable weights in the weighted sum over tokens in sequence
        # Number of tokens = max length of a sequence to be squashed into one token (?minus 2 special tokens?)
        self.w_tokens = nn.Parameter(torch.rand(int(args.max_length)))

        # Optional parameters for additional configuration
        self.special_tokens_to_zero = bool(not args.not_special_tokens_to_zero)
        self.layer_normalization_on = args.add_layer_pre_normalization
        # Normalize layer output
        self.optional_norm_1 = nn.LayerNorm(args.hidden_size) if self.layer_normalization_on else nn.Identity()
        self.optional_norm_2 = nn.LayerNorm(args.hidden_size) if self.layer_normalization_on else nn.Identity()

    def __call__(self, hidden_states, masks: torch.Tensor) -> torch.Tensor:
        x = zero_out_padding_and_special_tokens(hidden_states, masks, self.special_tokens_to_zero)
        x = torch.stack(x, dim=1)
        return x


class OneLayerWSumTokens(Combination):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_layer_to_use = args.hidden_layer_to_use

    def __call__(self, outputs, masks):
        one_layer_hidden_state = tuple(outputs.hidden_states[self.hidden_layer_to_use].unsqueeze(0))
        x = super().__call__(one_layer_hidden_state, masks)
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        x = x.squeeze(1)
        x = torch.einsum('bsh,s->bh', x, self.w_tokens)
        x = self.optional_norm_2(x) if self.layer_normalization_on else x
        return x


class OneLayerMaxPoolTokens(Combination):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_layer_to_use = args.hidden_layer_to_use

    def __call__(self, outputs, masks):
        one_layer_hidden_state = tuple(outputs.hidden_states[self.hidden_layer_to_use].unsqueeze(0))
        x = super().__call__(one_layer_hidden_state, masks)
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        x = x.squeeze(1)
        # dim = BSH
        x = torch.max(x, dim=1).values
        x = self.optional_norm_2(x) if self.layer_normalization_on else x
        return x


class OneLayerCLS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_layer_to_use = args.hidden_layer_to_use

    def __call__(self, outputs, masks):
        one_layer_hidden_state = outputs.hidden_states[self.hidden_layer_to_use]
        x = one_layer_hidden_state[:, 0, :]
        return x


class WSumTokensWSumLayers(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        x = super().__call__(outputs.hidden_states[1:], masks)
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        x = torch.einsum('blsh,s->blh', x, self.w_tokens)
        x = torch.einsum('blh,l->bh', x, self.w_layers)
        x = self.optional_norm_2(x) if self.layer_normalization_on else x
        return x


class MaxPoolLayersWSumTokens(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        hidden_states = super().__call__(outputs.hidden_states[1:], masks)
        # dim = [B L S H]
        max_over_layers = torch.max(hidden_states, dim=1).values
        # dim = BSH
        w_sum_over_tokens = torch.einsum('bsh,s->bh', max_over_layers, self.w_tokens)
        # dim = BH
        x = self.optional_norm_1(w_sum_over_tokens) if self.layer_normalization_on else w_sum_over_tokens
        # dim = BH
        return x


class WSumLayersWSumTokens(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        x = super().__call__(outputs.hidden_states[1:], masks)
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        x = torch.einsum('blsh,l->bsh', x, self.w_layers)
        x = torch.einsum('bsh,s->bh', x, self.w_tokens)
        x = self.optional_norm_2(x) if self.layer_normalization_on else x
        return x


class MaxPoolTokensWSumLayers(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        hidden_states = super().__call__(outputs.hidden_states[1:], masks)
        # dim = [B L S H]
        max_over_tokens = torch.max(hidden_states, dim=2).values
        # dim = BLH
        w_sum_over_layers = torch.einsum('blh,s->bh', max_over_tokens, self.w_layers)
        # dim = BH
        x = self.optional_norm_1(w_sum_over_layers) if self.layer_normalization_on else w_sum_over_layers
        # dim = BH
        return x


class MaxPoolLayersMaxPoolTokens(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        hidden_states = super().__call__(outputs.hidden_states[1:], masks)
        # dim = [B L S H]
        max_over_layers = torch.max(hidden_states, dim=1).values
        # dim = BSH
        max_over_tokens = torch.max(max_over_layers, dim=1).values
        # dim = BH
        x = self.optional_norm_1(max_over_tokens) if self.layer_normalization_on else max_over_tokens
        # dim = BH
        return x


class WSumCLS(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        # dim = [B (L+1) S H]
        cls_tokens = hidden_states[:, 1:, 0, :]
        # dim = LBH
        x = torch.einsum('blh,l->bh', cls_tokens, self.w_layers)
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        return x


class MaxPoolCLS(Combination):
    def __init__(self, args):
        super().__init__(args)

    def __call__(self, outputs, masks):
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        # dim = [B (L+1) S H]
        cls_tokens = hidden_states[:, 1:, 0, :]
        # dim = BLH
        x = torch.max(cls_tokens, dim=1).values
        # dim = BH
        x = self.optional_norm_1(x) if self.layer_normalization_on else x
        return x


class LastLayerCLS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clf_architecture = args.clf_architecture

    def __call__(self, outputs, masks):
        # return all cls tokens of the batch, dim = BH
        # we don't use layer normalization here, because operations are just slicing
        if self.clf_architecture == 'one_linear_layer':
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state[:, 0, :]
