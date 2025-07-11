# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerFlatten(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,printsize=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, hw = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)
        # print("src",src.shape)
        # print("mask",mask.shape)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed,printsize=printsize)
        # print("tgt",tgt.shape)
        # print("memory",memory.shape,memory.permute(1, 2, 0).view(bs, c, h, w).shape)
        # print("mask",mask.shape)
        # print("pos_embed",pos_embed.shape)
        # print("query_embed",query_embed.shape)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed,printsize=printsize)
        # print("hs",hs.shape)
        # print("----------------------")
        return hs.transpose(1, 2), memory.permute(1, 2, 0)

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,printsize=False):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        # print("src",src.shape)
        # print("mask",mask.shape)
        tgt = torch.zeros_like(query_embed) # num, batch_size, dim
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed,printsize=printsize)
        # print("tgt",tgt.shape)
        # print("memory",memory.shape,memory.permute(1, 2, 0).view(bs, c, h, w).shape)
        # print("mask",mask.shape)
        # print("pos_embed",pos_embed.shape)
        # print("query_embed",query_embed.shape)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed,printsize=printsize)
        # print("hs",hs.shape)
        # print("----------------------")
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        output = src

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos, printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_sigmoid_weights:
            return output, sigmoid_weights
        else:
            return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        output = tgt

        intermediate = []

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output, memory, tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               pos=pos, query_pos=query_pos, printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if need_sigmoid_weights:
            return output.unsqueeze(0),sigmoid_weights
        else:
            return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        q = k = self.with_pos_embed(src, pos)
        if printsize:
            print(163, torch.sort(q.flatten())[0])
            print(164, torch.sort(k.flatten())[0])
            print(165, torch.sort(src.flatten())[0])
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        if printsize:
            print(166, torch.sort(src.flatten())[0])
        src = self.norm1(src)
        if printsize:
            print(168,torch.sort(src.flatten())[0])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if printsize:
            print(171, torch.sort(src.flatten())[0])
        src = self.norm2(src)
        if printsize:
            print(173, torch.sort(src.flatten())[0])
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,need_sigmoid_weights=False):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos,need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        q = k = self.with_pos_embed(tgt, query_pos)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        if printsize:
            print(235, torch.sort(tgt.flatten())[0])
        tgt = self.norm1(tgt)
        if printsize:
            print(237, torch.sort(tgt.flatten())[0])
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if printsize:
            print(243, torch.sort(tgt.flatten())[0])
        tgt = self.norm2(tgt)
        if printsize:
            print(245, torch.sort(tgt.flatten())[0])
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if printsize:
            print(248, torch.sort(tgt.flatten())[0])
        tgt = self.norm3(tgt)
        if printsize:
            print(250, torch.sort(tgt.flatten())[0])
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,need_sigmoid_weights=False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)


class CrossTransformerEncoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,real_src,gt_src,
                     gt_src_mask: Optional[Tensor] = None,
                     gt_src_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):

        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(real_src, gt_src, value=gt_src,
                                                 attn_mask=gt_src_mask,
                                                 key_padding_mask=gt_src_key_padding_mask,
                                                 need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2,weight = self.self_attn(real_src, gt_src, value=gt_src,
                                         attn_mask=gt_src_mask,
                                         key_padding_mask=gt_src_key_padding_mask,
                                         need_sigmoid_weights=need_sigmoid_weights)
        src = real_src + self.dropout1(src2)
        # print("src",src.shape)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

class CrossTransformerEncoder3(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, real_src,gt_src,
                real_src_mask: Optional[Tensor] = None,
                real_src_key_padding_mask: Optional[Tensor] = None,
                gt_src_mask: Optional[Tensor] = None,
                gt_src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,printsize=False,need_sigmoid_weights=False):
        output = real_src

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output,gt_src,
                                               real_src_mask=real_src_mask,real_src_key_padding_mask=real_src_key_padding_mask,
                                               gt_src_mask=gt_src_mask,gt_src_key_padding_mask=gt_src_key_padding_mask,
                                               pos=pos, printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output,gt_src,
                               real_src_mask=real_src_mask,real_src_key_padding_mask=real_src_key_padding_mask,
                               gt_src_mask=gt_src_mask,gt_src_key_padding_mask=gt_src_key_padding_mask,
                               pos=pos,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_sigmoid_weights:
            return output, sigmoid_weights
        else:
            return output

class CrossTransformerEncoderLayer3(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.other_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,real_src,gt_src,
                     real_src_mask: Optional[Tensor] = None,
                     real_src_key_padding_mask: Optional[Tensor] = None,
                     gt_src_mask: Optional[Tensor] = None,
                     gt_src_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):

        if need_sigmoid_weights:
            real_src2, real_sigmoid_weight = self.self_attn(real_src, real_src, value=real_src,
                                                  attn_mask=real_src_mask,
                                                  key_padding_mask=real_src_key_padding_mask,
                                                  need_sigmoid_weights=need_sigmoid_weights)
            src2,sigmoid_weight = self.other_attn(real_src, gt_src, value=gt_src,
                                                 attn_mask=gt_src_mask,
                                                 key_padding_mask=gt_src_key_padding_mask,
                                                 need_sigmoid_weights=need_sigmoid_weights)
        else:
            real_src2,real_weight = self.self_attn(real_src, real_src, value=real_src,
                                      attn_mask=real_src_mask,
                                      key_padding_mask=real_src_key_padding_mask,
                                      need_sigmoid_weights=need_sigmoid_weights)
            src2,weight = self.other_attn(query=real_src, key=gt_src, value=gt_src,
                                         attn_mask=gt_src_mask,
                                         key_padding_mask=gt_src_key_padding_mask,
                                         need_sigmoid_weights=need_sigmoid_weights)

        src = real_src + real_src2 + self.dropout1(src2)
        # print("src",src.shape)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

class GroupTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = GroupTransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = GroupTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = GroupTransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = GroupTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sub_src, sub_key_padding_mask, sub_gt_key_padding_mask, sub_tgt,
                obj_src, obj_key_padding_mask, obj_gt_key_padding_mask, obj_tgt,
                printsize=False):
        # src: max_len, phrase, feature_dim
        # mask: phrase, max_len
        # tgt: max_len, phrase, feature_dim
        sub_memory = self.encoder(sub_src,src_key_padding_mask=sub_key_padding_mask,printsize=printsize)
        obj_memory = self.encoder(obj_src,src_key_padding_mask=obj_key_padding_mask,printsize=printsize)

        if obj_gt_key_padding_mask is None:
            sub_hs = self.decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_key_padding_mask,
                                  printsize=printsize)
        else:
            sub_hs = self.decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask,
                                  other_memory_key_padding_mask=obj_gt_key_padding_mask,
                                  printsize=printsize)
        if sub_gt_key_padding_mask is None:
            obj_hs = self.decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_key_padding_mask,
                                  printsize=printsize)
        else:
            obj_hs = self.decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_gt_key_padding_mask,
                                  printsize=printsize)
        # print("hs",hs.shape)
        # print("----------------------")
        return sub_hs, obj_hs

class GroupTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        output = src

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output, src_attn_mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask,
                                               printsize=printsize, need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output, src_attn_mask=src_attn_mask, src_key_padding_mask=src_key_padding_mask,
                               printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_sigmoid_weights:
            return output, sigmoid_weights
        else:
            return output

class GroupTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, other_memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                other_memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                other_memory_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        output = tgt

        intermediate = []

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output, memory, other_memory,
                               tgt_attn_mask=tgt_attn_mask,
                               memory_attn_mask=memory_attn_mask,
                               other_memory_attn_mask=other_memory_attn_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               other_memory_key_padding_mask=other_memory_key_padding_mask,
                               printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output, memory, other_memory,
                               tgt_attn_mask=tgt_attn_mask,
                               memory_attn_mask=memory_attn_mask,
                               other_memory_attn_mask=other_memory_attn_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               other_memory_key_padding_mask=other_memory_key_padding_mask,
                               printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if need_sigmoid_weights:
            return output,sigmoid_weights
        else:
            return output

class GroupTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,
                    src_attn_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    need_sigmoid_weights=False):
        src2 = self.norm1(src)
        q = k = src2
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(q, k, value=src2, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward_post(self,src,
                     src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        q = k = src
        if printsize:
            print(163, torch.sort(q.flatten())[0])
            print(164, torch.sort(k.flatten())[0])
            print(165, torch.sort(src.flatten())[0])
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(q, k, value=src, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        if printsize:
            print(166, torch.sort(src.flatten())[0])
        src = self.norm1(src)
        if printsize:
            print(168,torch.sort(src.flatten())[0])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if printsize:
            print(171, torch.sort(src.flatten())[0])
        src = self.norm2(src)
        if printsize:
            print(173, torch.sort(src.flatten())[0])
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward(self, src,
                src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(src, src_attn_mask, src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(src, src_attn_mask, src_key_padding_mask,printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

class GroupTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.other_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        if printsize:
            print(708,tgt)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        if printsize:
            print(715,tgt2)
        tgt = tgt + self.dropout1(tgt2)
        if printsize:
            print(235, torch.sort(tgt.flatten())[0])
        tgt = self.norm1(tgt)
        if printsize:
            print(237, torch.sort(tgt.flatten())[0])
        tgt2 = self.other_attn(query=tgt,
                                   key=other_memory,
                                   value=other_memory,
                                   attn_mask=other_memory_attn_mask,
                                   key_padding_mask=other_memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if printsize:
            print(243, torch.sort(tgt.flatten())[0])
        tgt = self.norm2(tgt)
        if printsize:
            print(245, torch.sort(tgt.flatten())[0])
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if printsize:
            print(248, torch.sort(tgt.flatten())[0])
        tgt = self.norm3(tgt)
        if printsize:
            print(250, torch.sort(tgt.flatten())[0])
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward_pre(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        tgt2 = self.norm1(tgt)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.other_attn(query=tgt2,
                                   key=other_memory,
                                   value=other_memory, attn_mask=other_memory_attn_mask,
                                   key_padding_mask=other_memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, other_memory,
                                    tgt_attn_mask, memory_attn_mask,other_memory_attn_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, other_memory_key_padding_mask,
                                    need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(tgt, memory, other_memory,
                                 tgt_attn_mask, memory_attn_mask,other_memory_attn_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, other_memory_key_padding_mask,
                                 printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)


class PhraseTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = PhraseTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = PhraseTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        predicate_decoder_layer = PhraseTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        predicate_decoder_norm = nn.LayerNorm(d_model)
        self.predicate_decoder = PhraseTransformerDecoder(predicate_decoder_layer, num_decoder_layers, predicate_decoder_norm, return_intermediate=return_intermediate_dec)

        group_decoder_layer = PhraseTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        group_decoder_norm = nn.LayerNorm(d_model)
        self.group_decoder = PhraseTransformerDecoder(group_decoder_layer, num_decoder_layers, group_decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sub_src, sub_key_padding_mask, sub_gt_key_padding_mask,sub_obj_tgt, sub_tgt, # phrase, 96, 1024
                obj_src, obj_key_padding_mask, obj_gt_key_padding_mask,obj_sub_tgt, obj_tgt, # phrase, 96, 1024
                printsize=False):
        # src: max_len, phrase, feature_dim
        # mask: phrase, max_len
        # tgt: max_len, phrase, feature_dim

        if obj_gt_key_padding_mask is None:
            sub_memory = self.encoder(sub_src,obj_src,src_key_padding_mask=sub_key_padding_mask,other_src_key_padding_mask=obj_key_padding_mask,printsize=printsize)
        else:
            sub_memory = self.encoder(sub_src, obj_src, src_key_padding_mask=sub_key_padding_mask,other_src_key_padding_mask=obj_gt_key_padding_mask, printsize=printsize)
        if sub_gt_key_padding_mask is None:
            obj_memory = self.encoder(obj_src,sub_src,src_key_padding_mask=obj_key_padding_mask,other_src_key_padding_mask=sub_key_padding_mask,printsize=printsize)
        else:
            obj_memory = self.encoder(obj_src, sub_src, src_key_padding_mask=obj_key_padding_mask, other_src_key_padding_mask=sub_gt_key_padding_mask, printsize=printsize)

        if obj_gt_key_padding_mask is None:
            sub_obj_predicate_embed = self.predicate_decoder(sub_obj_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_key_padding_mask,
                                  printsize=printsize)
        else:
            sub_obj_predicate_embed = self.predicate_decoder(sub_obj_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_gt_key_padding_mask,
                                  printsize=printsize)
        if sub_gt_key_padding_mask is None:
            obj_sub_predicate_embed = self.predicate_decoder(obj_sub_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_key_padding_mask,
                                  printsize=printsize)
        else:
            obj_sub_predicate_embed = self.predicate_decoder(obj_sub_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_gt_key_padding_mask,
                                  printsize=printsize)

        if obj_gt_key_padding_mask is None:
            sub_embed = self.group_decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_key_padding_mask,
                                  printsize=printsize)
        else:
            sub_embed = self.group_decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_gt_key_padding_mask,
                                  printsize=printsize)
        if sub_gt_key_padding_mask is None:
            obj_embed = self.group_decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_key_padding_mask,
                                  printsize=printsize)
        else:
            obj_embed = self.group_decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_gt_key_padding_mask,
                                  printsize=printsize)

        return sub_obj_predicate_embed, obj_sub_predicate_embed, sub_embed, obj_embed

class PhraseTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, other_src,
                src_attn_mask: Optional[Tensor] = None,
                other_src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                other_src_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        output = other_src

        for layer in self.layers:
            if need_sigmoid_weights:
                output, sigmoid_weights = layer(src, output,
                                                src_attn_mask=src_attn_mask,
                                                other_src_attn_mask=other_src_attn_mask,
                                                src_key_padding_mask=src_key_padding_mask,
                                                other_src_key_padding_mask = other_src_key_padding_mask,
                                                printsize=printsize, need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(src, output,
                               src_attn_mask=src_attn_mask,
                               other_src_attn_mask=other_src_attn_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               other_src_key_padding_mask=other_src_key_padding_mask,
                               printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_sigmoid_weights:
            return output, sigmoid_weights
        else:
            return output

class PhraseTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, other_memory,
                tgt_attn_mask: Optional[Tensor] = None,
                memory_attn_mask: Optional[Tensor] = None,
                other_memory_attn_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                other_memory_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        output = tgt

        intermediate = []

        for layer in self.layers:
            if need_sigmoid_weights:
                output,sigmoid_weights = layer(output, memory, other_memory,
                               tgt_attn_mask=tgt_attn_mask,
                               memory_attn_mask=memory_attn_mask,
                               other_memory_attn_mask=other_memory_attn_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               other_memory_key_padding_mask=other_memory_key_padding_mask,
                               printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            else:
                output = layer(output, memory, other_memory,
                               tgt_attn_mask=tgt_attn_mask,
                               memory_attn_mask=memory_attn_mask,
                               other_memory_attn_mask=other_memory_attn_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               other_memory_key_padding_mask=other_memory_key_padding_mask,
                               printsize=printsize,
                               need_sigmoid_weights=need_sigmoid_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if need_sigmoid_weights:
            return output,sigmoid_weights
        else:
            return output

class PhraseTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src,other_src,
                    src_attn_mask: Optional[Tensor] = None,
                    other_src_attn_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    other_src_key_padding_mask: Optional[Tensor] = None,
                    need_sigmoid_weights=False):
        src2 = self.norm1(src)
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(src2, other_src, value=other_src, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(src2, other_src, value=other_src, attn_mask=src_attn_mask,
                                  key_padding_mask=src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward_post(self,src,other_src,
                     src_attn_mask: Optional[Tensor] = None,
                     other_src_attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     other_src_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        if need_sigmoid_weights:
            src2,sigmoid_weight = self.self_attn(src, other_src, value=other_src, attn_mask=other_src_attn_mask,
                                  key_padding_mask=other_src_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            src2 = self.self_attn(src, other_src, value=other_src, attn_mask=other_src_attn_mask,
                                  key_padding_mask=other_src_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)[0]
        src = src + self.dropout1(src2)
        if printsize:
            print(166, torch.sort(src.flatten())[0])
        src = self.norm1(src)
        if printsize:
            print(168,torch.sort(src.flatten())[0])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        if printsize:
            print(171, torch.sort(src.flatten())[0])
        src = self.norm2(src)
        if printsize:
            print(173, torch.sort(src.flatten())[0])
        if need_sigmoid_weights:
            return src,sigmoid_weight
        else:
            return src

    def forward(self, src, other_src,
                src_attn_mask: Optional[Tensor] = None,
                other_src_attn_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                other_src_key_padding_mask: Optional[Tensor] = None,
                printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(src, other_src, src_attn_mask, other_src_attn_mask, src_key_padding_mask, other_src_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(src, other_src, src_attn_mask, other_src_attn_mask, src_key_padding_mask, other_src_key_padding_mask, printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

class PhraseTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.other_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        if printsize:
            print(708,tgt)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        if printsize:
            print(715,tgt2)
        tgt = tgt + self.dropout1(tgt2)
        if printsize:
            print(235, torch.sort(tgt.flatten())[0])
        tgt = self.norm1(tgt)
        if printsize:
            print(237, torch.sort(tgt.flatten())[0])
        tgt2 = self.other_attn(query=tgt,
                                   key=memory,
                                   value=memory,
                                   attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        if printsize:
            print(243, torch.sort(tgt.flatten())[0])
        tgt = self.norm2(tgt)
        if printsize:
            print(245, torch.sort(tgt.flatten())[0])
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        if printsize:
            print(248, torch.sort(tgt.flatten())[0])
        tgt = self.norm3(tgt)
        if printsize:
            print(250, torch.sort(tgt.flatten())[0])
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward_pre(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        tgt2 = self.norm1(tgt)
        if need_sigmoid_weights:
            tgt2,sigmoid_weights = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask, need_sigmoid_weights=need_sigmoid_weights)
        else:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_attn_mask,
                                  key_padding_mask=tgt_key_padding_mask,need_sigmoid_weights=need_sigmoid_weights)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.other_attn(query=tgt2,
                                   key=memory,
                                   value=memory, attn_mask=memory_attn_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if need_sigmoid_weights:
            return tgt,sigmoid_weights
        else:
            return tgt

    def forward(self, tgt, memory, other_memory,
                     tgt_attn_mask: Optional[Tensor] = None,
                     memory_attn_mask: Optional[Tensor] = None,
                     other_memory_attn_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     other_memory_key_padding_mask: Optional[Tensor] = None,
                     printsize=False,need_sigmoid_weights=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, other_memory,
                                    tgt_attn_mask, memory_attn_mask,other_memory_attn_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, other_memory_key_padding_mask,
                                    need_sigmoid_weights=need_sigmoid_weights)
        return self.forward_post(tgt, memory, other_memory,
                                 tgt_attn_mask, memory_attn_mask,other_memory_attn_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, other_memory_key_padding_mask,
                                 printsize=printsize,need_sigmoid_weights=need_sigmoid_weights)

class GroupTransformer2(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = PhraseTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = PhraseTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        predicate_decoder_layer = PhraseTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        predicate_decoder_norm = nn.LayerNorm(d_model)
        self.predicate_decoder = PhraseTransformerDecoder(predicate_decoder_layer, num_decoder_layers, predicate_decoder_norm, return_intermediate=return_intermediate_dec)

        group_decoder_layer = PhraseTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        group_decoder_norm = nn.LayerNorm(d_model)
        self.group_decoder = PhraseTransformerDecoder(group_decoder_layer, num_decoder_layers, group_decoder_norm, return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sub_src, sub_key_padding_mask, sub_gt_key_padding_mask,sub_obj_tgt, sub_tgt, # phrase, 96, 1024
                obj_src, obj_key_padding_mask, obj_gt_key_padding_mask,obj_sub_tgt, obj_tgt, # phrase, 96, 1024
                printsize=False):
        # src: max_len, phrase, feature_dim
        # mask: phrase, max_len
        # tgt: max_len, phrase, feature_dim

        sub_memory = self.encoder(sub_src,obj_src,src_key_padding_mask=sub_key_padding_mask,other_src_key_padding_mask=obj_key_padding_mask,printsize=printsize)
        obj_memory = self.encoder(obj_src,sub_src,src_key_padding_mask=obj_key_padding_mask,other_src_key_padding_mask=sub_key_padding_mask,printsize=printsize)

        if obj_gt_key_padding_mask is None:
            sub_embed = self.group_decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_key_padding_mask,
                                  printsize=printsize)
        else:
            sub_embed = self.group_decoder(sub_tgt,
                                  memory=sub_memory, other_memory=obj_memory,
                                  memory_key_padding_mask=sub_key_padding_mask, other_memory_key_padding_mask=obj_gt_key_padding_mask,
                                  printsize=printsize)
        if sub_gt_key_padding_mask is None:
            obj_embed = self.group_decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_key_padding_mask,
                                  printsize=printsize)
        else:
            obj_embed = self.group_decoder(obj_tgt,
                                  memory=obj_memory, other_memory=sub_memory,
                                  memory_key_padding_mask=obj_key_padding_mask, other_memory_key_padding_mask=sub_gt_key_padding_mask,
                                  printsize=printsize)

        return sub_embed, obj_embed

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.modules.linear._LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.q_proj_weight)
            torch.nn.init.xavier_uniform_(self.k_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            torch.nn.init.constant_(self.in_proj_bias, 0.)
            torch.nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            torch.nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            torch.nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, need_sigmoid_weights=False, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_sigmoid_weights=need_sigmoid_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights, need_sigmoid_weights=need_sigmoid_weights,
                attn_mask=attn_mask)

import warnings
def multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 need_sigmoid_weights: bool = False,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None
                                 ):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and F.has_torch_function(tens_ops):
            return F.handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    softmax_attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    softmax_attn_output_weights = F.dropout(softmax_attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(softmax_attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        softmax_attn_output_weights = softmax_attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, softmax_attn_output_weights.sum(dim=1) / num_heads
    elif need_sigmoid_weights:
        sigmoid_attn_output_weights=F.sigmoid(attn_output_weights,dim=-1)
        sigmoid_attn_output_weights = sigmoid_attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, sigmoid_attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, attn_output_weights