from typing import Dict, Tuple, List

import torch
from torch import nn

from cat_sam.models.module_lib import MLP, LayerNorm2d
from cat_sam.models.segment_anything_ext.modeling import TwoWayTransformer, MaskDecoder


class MaskDecoderHQ(MaskDecoder):
    """
    Adopted from Sam-HQ:
    https://github.com/SysCV/sam-hq/blob/322488826bda616798901c6280d13a9a90444ae7/train/train.py#L67

    """
    def __init__(self, model_type: str, sam_decoder_state_dict: Dict):
        super().__init__(transformer_dim=256,
                         transformer=TwoWayTransformer(
                             depth=2,
                             embedding_dim=256,
                             mlp_dim=2048,
                             num_heads=8,
                         ),
                         num_multimask_outputs=3,
                         activation=nn.GELU,
                         iou_head_depth=3,
                         iou_head_hidden_dim=256, )
        assert model_type in ["vit_b", "vit_l", "vit_h"]
        self.load_state_dict(sam_decoder_state_dict)
        for n, p in self.named_parameters():
            p.requires_grad = False
        self.froze_modules = [n for n, _ in self.named_children()]
        self.froze_params = [n for n, _ in self.named_parameters()]

        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))

    def train(self, mode: bool = True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: turn the modules of original SAM mask decoder to eval mode
            for n, c in self.named_children():
                if n in self.froze_modules:
                    c.eval()
                else:
                    c.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            interm_embeddings: torch.Tensor,
            hq_token_weight: torch.Tensor = None,
            return_all_hq_masks: bool = False
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        if isinstance(self.compress_vit_feat, List):
            hq_features = self.embedding_encoder(image_embeddings)
            for i in range(len(self.compress_vit_feat)):
                vit_features = interm_embeddings[i].permute(0, 3, 1, 2)
                hq_features += self.compress_vit_feat[i](vit_features)
        # for compatibility with the original SAM-HQ ckpt
        else:
            vit_features = interm_embeddings[0].permute(0, 3, 1,
                                                        2)  # early-layer ViT feature, after 1st global attention block in ViT
            hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        batch_size = len(image_embeddings)
        masks_sam_batch, masks_hq_batch = [], []
        for i_batch in range(batch_size):
            masks, iou_preds = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature=hq_features[i_batch].unsqueeze(0),
                hq_token_weight=hq_token_weight
            )

            # Select the correct mask or masks for output
            if multimask_output:
                # mask with the highest score
                mask_slice = slice(1, self.num_mask_tokens - 1)
                iou_preds = iou_preds[:, mask_slice]
                iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
                masks_multi = masks[:, mask_slice, :, :]
                masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
            else:
                # singale mask output, default
                mask_slice = slice(0, 1)
                masks_sam = masks[:, mask_slice]

            masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :]
            masks_sam_batch.append(masks_sam)
            masks_hq_batch.append(masks_hq)
        return masks_sam_batch, masks_hq_batch

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            hq_feature: torch.Tensor,
            hq_token_weight: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        if hq_token_weight is None:
            hq_token_weight = self.hf_token.weight
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, hq_token_weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:, :4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:, 4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_ours], dim=1)

        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred