import torch
from torch import nn

from cat_sam.models.module_lib import Adapter, PromptGenerator


class SAMImageEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMImageEncodeWrapper, self).__init__()
        self.sam_img_encoder = ori_sam.image_encoder
        if fix:
            for name, param in self.sam_img_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.sam_img_encoder(x)
        return x


class SAMPromptEncodeWrapper(nn.Module):

    def __init__(self, ori_sam, fix: bool = True):
        super(SAMPromptEncodeWrapper, self).__init__()
        self.sam_prompt_encoder = ori_sam.prompt_encoder
        if fix:
            for name, param in self.sam_prompt_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points, boxes, masks)
        return sparse_embeddings, dense_embeddings

    def get_dense_pe(self):
        return self.sam_prompt_encoder.get_dense_pe()


class CATSAMTImageEncoder(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,
    ):
        super(CATSAMTImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)
        self.hq_token = hq_token

        total_p_layer = len(self.sam_img_encoder.blocks)
        prompt_dim = self.sam_img_encoder.pos_embed.shape[-1]
        self.hq_token_proj = nn.Sequential(
            *[Adapter(hq_token.size(-1), prompt_dim, mlp_ratio=0.25) for _ in range(total_p_layer)]
        )


    def forward(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        hq_prompt_tokens = []
        for i in range(0, len(self.hq_token_proj)):
            hq_prompt_tokens.append(self.hq_token_proj[i](self.hq_token).unsqueeze(0))

        interm_embeddings = []
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x, hq_prompt_tokens[i])
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings



class CATSAMAImageEncoder(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor
    ):
        super(CATSAMAImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)

        self.prompt_generator = PromptGenerator(
            scale_factor=32, prompt_type='highpass',
            embed_dim=self.sam_img_encoder.patch_embed.proj.out_channels,
            tuning_stage=1234, depth=len(self.sam_img_encoder.blocks), input_type='fft', freq_nums=0.25,
            handcrafted_tune=True, embedding_tune=True, adaptor='adaptor',
            img_size=self.sam_img_encoder.img_size,
            patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        )

        self.hq_token = hq_token
        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125, add_last_layer=False)
              for _ in range(self.prompt_generator.shared_mlp.in_features)]
        )

        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(patch_height * patch_width)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        embedding_feature = self.prompt_generator.init_embeddings(x)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        hq_feature = torch.cat(
            [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],
            dim=-1
        )
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature, hq_feature=hq_feature)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings
