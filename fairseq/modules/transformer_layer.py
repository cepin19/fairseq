# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import math
from typing import Dict, List, Optional
import collections
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, RelPartialLearnableMultiHeadAttn
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
import torch.nn.functional as F
import copy
import logging
def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):
    logging.debug(seq_len)
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
       # logging.info(max_len)
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        #broad_cast_seq_len = x.new_zeros(x.shape)
        #logging.info("bc_seq_len")
        #logging.info(broad_cast_seq_len)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb
class MultiHead_Attention_Lattice_rel(nn.Module):
    def __init__(self, hidden_size, num_heads, pe,
                 pe_ss,pe_se,pe_es,pe_ee,
                 scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None):
        '''
        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        self.mode="debug"
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        if self.mode['debug']:
            logging.debug('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4),
                                                nn.Softmax(dim=-1))
            print('暂时不支持以gate融合pos信息')
            exit(1208)


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = nn.Dropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len, lex_num, pos_s,pos_e,attn_mask=None):
        batch = key.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)

        if self.mode['debug']:
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))
            print('pos_ss:{}'.format(pos_ss))
            print('pos_se:{}'.format(pos_se))
            print('pos_es:{}'.format(pos_es))
            print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = key.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        pe_ss = self.pe[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        pe_se = self.pe[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())


        # E prepare relative position encoding

        if self.k_proj:
            if self.mode['debug']:
                logging.debug('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                logging.debug('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                logging.debug('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                logging.debug('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch,max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)


        #A
        A_ = torch.matmul(query,key)

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        #C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        key_for_c = key
        C_ = torch.matmul(u_for_c, key)

        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            logging.debug('A:{}'.format(A_.size()))
            logging.debug('B:{}'.format(B_.size()))
            logging.debug('C:{}'.format(C_.size()))
            logging.debug('D:{}'.format(D_.size()))
        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)
     #   logging.info(seq_len)
        mask = seq_len_to_mask(seq_len).bool().unsqueeze(1).unsqueeze(1)#.cuda()
      #  logging.info(mask.shape)
       # logging.info(attn_score_raw.shape)

        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        logging.debug("attn_score_raw_masked")
        logging.debug(attn_score_raw_masked.shape)

        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.formaregistt(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            #print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''
        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rel_pos_init=0
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.lattice=getattr(args, 'lattice', False)
        if self.lattice:
            self.self_attn = self.build_self_attention_lattice(self.embed_dim, args)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention_lattice(self, embed_dim, args):
        self.learnable_position=False
        pe = get_embedding(384, embed_dim/2, rel_pos_init=self.rel_pos_init)
        self.pos_fusion=nn.Sequential(nn.Linear(embed_dim * 2, embed_dim),
                           nn.ReLU(inplace=True))
        self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
        self.pe_ss = pe# nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        self.pe_se = pe#nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        self.pe_es = pe#nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        self.pe_ee = pe#nn.Parameter(copy.deepcopy(pe), requires_grad=self.learnable_position)
        #return RelPartialLearnableMultiHeadAttn(args.encoder_attention_heads,embed_dim,embed_dim//args.encoder_attention_heads,0.1)
        return MultiHead_Attention_Lattice_rel(embed_dim,args.encoder_attention_heads,self.pe,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,attn_dropout=0.1,four_pos_fusion="ff")
        #return MultiheadAttention(
        #    embed_dim,
        #    args.encoder_attention_heads,
        #    dropout=args.attention_dropout,
        #    self_attention=True,
        #    q_noise=self.quant_noise,
        #    qn_block_size=self.quant_noise_block_size,
        #)
    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, pos_s=0, pos_e=0):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        #x, _ = self.self_attn(
        #    query=x,
        #    key=x,
        #    value=x,
        #    key_padding_mask=encoder_padding_mask,
        #    attn_mask=attn_mask,
        #)
        if self.lattice:
            logging.debug("pos_s")
            logging.debug(pos_s.shape)
            logging.debug(pos_e.shape)
            pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
            pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
            pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
            pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)


            logging.debug("pos_ss")
            logging.debug(pos_ss.shape)
            logging.debug(pos_ss)
            logging.debug(pos_ss.view(-1).shape)
            logging.debug("pe")
            logging.debug(self.pe.shape)
            batch=pos_s.size(0)
            max_seq_len=pos_s.size(1)
            pe_ss = self.pe[pos_ss.view(-1)].view(size=[batch,max_seq_len,max_seq_len,-1])
            pe_se = self.pe[(pos_se).view(-1)].view(size=[batch, max_seq_len, max_seq_len, -1])
            pe_es = self.pe[(pos_es).view(-1)].view(size=[batch, max_seq_len, max_seq_len, -1])
            pe_ee = self.pe[(pos_ee).view(-1)].view(size=[batch, max_seq_len, max_seq_len, -1])
            logging.debug("pe_ss")
            logging.debug(pe_ss.shape)
            logging.debug(pe_ss)

            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            logging.debug("x")

            logging.debug(x.shape)
            r= self.pos_fusion (pe_4).reshape([max_seq_len,batch,max_seq_len,-1])

            logging.debug("r")
            logging.debug(r.shape)

            #x  = self.self_attn(x,r)
            x=x.contiguous().view(batch, max_seq_len, -1)
            seq_len=torch.max(pos_s,1)[0]-1

            x  = self.self_attn(x,x,x,seq_len,[0],pos_s,pos_e).view(max_seq_len,batch,-1)
            logging.debug("OUTX")
            logging.debug(x.shape)
            logging.debug(x)
        else:
            x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
         )
     #       logging.info("OUTX")
      #      logging.info(x.shape)
        #key, query, value, seq_len, lex_num, pos_s, pos_e
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
