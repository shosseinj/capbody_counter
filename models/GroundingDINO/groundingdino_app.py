# Modified in 2026 by CapBody Counter contributors for local inference compatibility and simplification.
# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from groundingdino.util import get_tokenlizer
from groundingdino.util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list
from .bertwarper import BertModelWarper, generate_masks_with_special_tokens_and_transfer_map
from .utils import MLP, ContrastiveEmbed


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        load_text_encoder_weights=True,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # visual exemplar cropping
        self.feature_map_proj = nn.Conv2d((256 + 512 + 1024), hidden_dim, kernel_size=1)

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type, load_weights=load_text_encoder_weights)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(
            self.bert.config.hidden_size, self.hidden_dim, bias=True
        )
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(
            ["[CLS]", "[SEP]", ".", "?"]
        )

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [
            _class_embed for i in range(transformer.num_decoder_layers)
        ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def add_exemplar_tokens(self, tokenized, text_dict, exemplar_tokens, labels):
        input_ids = tokenized["input_ids"]

        device = input_ids.device
        new_input_ids = []
        encoded_text = text_dict["encoded_text"]
        new_encoded_text = []
        text_token_mask = text_dict["text_token_mask"]
        new_text_token_mask = []
        position_ids = text_dict["position_ids"]
        text_self_attention_masks = text_dict["text_self_attention_masks"]

        for sample_ind in range(len(labels)):
            label = labels[sample_ind][0]
            exemplars = exemplar_tokens[sample_ind]
            label_count = -1
            assert len(input_ids[sample_ind]) == len(position_ids[sample_ind])
            for token_ind in range(len(input_ids[sample_ind])):
                input_id = input_ids[sample_ind][token_ind]
                if (token_ind > 0 and input_ids[sample_ind][token_ind - 1] in self.specical_tokens):
                    label_count += 1
                if label_count == label:
                    # Get the index where to insert the exemplar tokens.
                    ind_to_insert_exemplar = token_ind
                    while (
                        input_ids[sample_ind][ind_to_insert_exemplar]
                        not in self.specical_tokens
                    ):
                        ind_to_insert_exemplar += 1
                    break

            # Handle no text case.
            if label_count == -1:
                ind_to_insert_exemplar = 1
            # * token indicates exemplar.
            new_input_ids.append(
                torch.cat(
                    [
                        input_ids[sample_ind][:ind_to_insert_exemplar],
                        torch.tensor([1008] * exemplars.shape[0]).to(device),
                        input_ids[sample_ind][ind_to_insert_exemplar:],
                    ]
                )
            )
            new_encoded_text.append(
                torch.cat(
                    [
                        encoded_text[sample_ind][:ind_to_insert_exemplar, :],
                        exemplars,
                        encoded_text[sample_ind][ind_to_insert_exemplar:, :],
                    ]
                )
            )
            new_text_token_mask.append(
                torch.full((len(new_input_ids[sample_ind]),), True).to(device)
            )

        tokenized["input_ids"] = torch.stack(new_input_ids)

        (
            text_self_attention_masks,
            position_ids,
            _,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, None
        )

        return ({
            "encoded_text": torch.stack(new_encoded_text),
            "text_token_mask": torch.stack(new_text_token_mask),
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }, tokenized)

    def combine_features(self, features):
        (bs, c, h, w) = (
            features[0].decompose()[0].shape[-4],
            features[0].decompose()[0].shape[-3],
            features[0].decompose()[0].shape[-2],
            features[0].decompose()[0].shape[-1],
        )

        x = torch.cat(
            [
                F.interpolate(
                    feat.decompose()[0],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                for feat in features
            ],
            dim=1,
        )

        x = self.feature_map_proj(x)

        return x

    def forward(
        self,
        input_images: NestedTensor,
        input_image_pos_exemplars: NestedTensor,
        pos_exemplars: List,
        input_images_neg_exemplars: List[NestedTensor],
        neg_exemplars: List,
        targets: List = None,
        cropped=False,
        orig_img=None,
        crop_width=0,
        crop_height=0,
        **kw,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            input_images.device
        )

        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][
                :, : self.max_text_len
            ]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][
                :, : self.max_text_len
            ]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v for k, v in tokenized.items() if k != "attention_mask"
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(
            bert_output["last_hidden_state"]
        )  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        if isinstance(input_images, (list, torch.Tensor)):
            input_images = nested_tensor_from_tensor_list(input_images)

        features, poss = self.backbone(input_images)

        # Positive exemplars
        features_pos_exemp, _ = self.backbone(input_image_pos_exemplars)
        combined_pos_features = self.combine_features(features_pos_exemp)
        # Get positive visual exemplar tokens.
        bs = len(pos_exemplars)
        num_pos_exemplars = pos_exemplars[0].shape[0]
        if num_pos_exemplars > 0:
            pos_exemplar_tokens = (
                roi_align(
                    combined_pos_features,
                    boxes=pos_exemplars,
                    output_size=(1, 1),
                    spatial_scale=(1 / 8),
                    aligned=True,
                )
                .squeeze(-1)
                .squeeze(-1)
                .reshape(bs, num_pos_exemplars, -1)
            )
        else:
            pos_exemplar_tokens = None

        if pos_exemplar_tokens is not None:
            text_dict, tokenized = self.add_exemplar_tokens(
                tokenized, text_dict, pos_exemplar_tokens, [torch.tensor([0]).to(input_images.device)]
            )

        # Negative exemplars
        num_neg_exemplar_imgs = len(input_images_neg_exemplars)
        for ind in range(num_neg_exemplar_imgs):
            exemplar_img = input_images_neg_exemplars[ind]
            exemplars = neg_exemplars[ind]
            bs = 1
            num_neg_exemplars = len(exemplars[0]) # Each set of exemplars has a batch dimension for compatibility with roialign
            if exemplar_img is not None and num_neg_exemplars > 0:
                features_neg_exemp, _ = self.backbone(exemplar_img)
                combined_neg_features = self.combine_features(features_neg_exemp)
                # Get negative visual exemplar tokens.
                neg_exemplar_tokens = (
                    roi_align(
                        combined_neg_features,
                        boxes=exemplars,
                        output_size=(1, 1),
                        spatial_scale=(1 / 8),
                        aligned=True,
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                    .reshape(bs, num_neg_exemplars, -1)
                )
                text_dict, tokenized = self.add_exemplar_tokens(
                    tokenized, text_dict, neg_exemplar_tokens, [torch.tensor([ind + 1]).to(input_images.device)]
                )

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = input_images.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
            "input_ids": tokenized["input_ids"],
        }
