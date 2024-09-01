import copy
from typing import Union, List, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from cat_sam.models.encoders import SAMImageEncodeWrapper, SAMPromptEncodeWrapper, CATSAMTImageEncoder, CATSAMAImageEncoder
from cat_sam.models.decoders import MaskDecoderHQ
from cat_sam.models.segment_anything_ext import sam_model_registry


sam_ckpt_path_dict = dict(
    vit_b='./pretrained/sam_vit_b_01ec64.pth',
    vit_l='./pretrained/sam_vit_l_0b3195.pth',
    vit_h='./pretrained/sam_vit_h_4b8939.pth'
)

class BaseCATSAM(nn.Module):

    def __init__(self, model_type: str):
        super(BaseCATSAM, self).__init__()
        assert model_type in ['vit_b', 'vit_l', 'vit_h'], f"invalid model_type: {model_type}!"
        self.ori_sam = sam_model_registry[model_type](sam_ckpt_path_dict[model_type])
        self.sam_img_size = (self.ori_sam.image_encoder.img_size, self.ori_sam.image_encoder.img_size)

        self.image_encoder = SAMImageEncodeWrapper(ori_sam=self.ori_sam, fix=True)
        self.prompt_encoder = SAMPromptEncodeWrapper(ori_sam=self.ori_sam, fix=True)

        self.mask_decoder = MaskDecoderHQ(
            model_type, self.ori_sam.mask_decoder.state_dict()
        )
        # remove the mask decoder in original SAM to avoid redundant params in model object
        del self.ori_sam.mask_decoder


    def train(self, mode: bool = True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training: only turn the image encoder to train mode
            for n, c in self.named_children():
                if n != 'image_encoder':
                    c.eval()
                else:
                    c.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)


    def forward(
            self,
            imgs: Union[List[torch.Tensor], None],
            point_coords: List[Union[torch.Tensor, None]],
            point_labels: List[Union[torch.Tensor, None]],
            box_coords: List[Union[torch.Tensor, None]],
            noisy_masks: List[Union[torch.Tensor, None]],
            image_embeddings: torch.Tensor = None,
            interm_embeddings: List[torch.Tensor] = None,
            ori_img_size: List[Tuple] = None,
            hq_token_weight: torch.Tensor = None,
            return_all_hq_masks: bool = False
    ):
        if (image_embeddings is None) ^ (interm_embeddings is None):
            raise RuntimeError("Please give image_embeddings and interm_embeddings at the same time for inference!")

        if image_embeddings is None and interm_embeddings is None:
            # record the original image size for later mask resizing
            ori_img_size = [(imgs[i].shape[-2], imgs[i].shape[-1]) for i in range(len(imgs))]
            imgs, point_coords, box_coords = self.preprocess(
                imgs=imgs, point_coords=point_coords, box_coords=box_coords, ori_img_size=ori_img_size)
            # imgs here is normalized with the shape of (B, 3, 1024, 1024)
            image_embeddings, interm_embeddings = self.image_encoder(imgs)
        elif ori_img_size is None:
            raise RuntimeError("Please also specify ori_img_size to forward() during the inference!")

        batch_size = len(image_embeddings)
        points, boxes, masks = self.convert_raw_prompts_to_triple(
            point_coords=point_coords, point_labels=point_labels,
            box_coords=box_coords, noisy_masks=noisy_masks, batch_size=batch_size
        )

        sparse_embeddings_list, dense_embeddings_list = [], []
        for batch_idx in range(batch_size):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points[batch_idx], boxes=boxes[batch_idx], masks=masks[batch_idx],
            )
            sparse_embeddings_list.append(sparse_embeddings)
            dense_embeddings_list.append(dense_embeddings)

        masks_sam, masks_hq = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=[self.prompt_encoder.get_dense_pe() for _ in range(batch_size)],
            sparse_prompt_embeddings=sparse_embeddings_list,
            dense_prompt_embeddings=dense_embeddings_list,
            multimask_output=False,
            interm_embeddings=interm_embeddings,
            hq_token_weight=hq_token_weight,
            return_all_hq_masks=return_all_hq_masks
        )

        # rescale the mask size back to original image size
        postprocess_masks_hq = [m_hq.clone() for m_hq in masks_hq]
        for i in range(len(postprocess_masks_hq)):
            postprocess_masks_hq[i]= self.postprocess(output_masks=postprocess_masks_hq[i], ori_img_size=ori_img_size[i])
        return postprocess_masks_hq


    @torch.no_grad()
    def set_infer_img(self, img) -> List[torch.Tensor]:
        if isinstance(img, torch.Tensor):
            if len(img.shape) <= 3:
                img = [img]
            elif len(img.shape) == 4:
                img = [i for i in img]
            else:
                raise RuntimeError
        elif not isinstance(img, List):
            raise RuntimeError

        for i in range(len(img)):
            if len(img[i].shape) == 2:
                img[i] = img[i].unsqueeze(0)
            if len(img[i].shape) == 3 and img[i].size(0) == 1:
                img[i] = img[i].repeat(3, 1, 1)
            if len(img[i].shape) != 3 or img[i].size(0) != 3:
                raise RuntimeError(
                    f'Wrong image shape! Each image in your given list must be either (H, W) or (3, H, W), '
                    f'but got {img[i].shape}!'
                )

        self.ori_infer_img_size = [(img[i].shape[-2], img[i].shape[-1]) for i in range(len(img))]
        self.ori_infer_img = img

        img, _, _ = self.preprocess(imgs=img, ori_img_size=self.ori_infer_img_size)
        self.img_features, self.interm_features = self.image_encoder(img)
        return img


    @torch.no_grad()
    def infer(
            self,
            point_coords: List[Union[List, torch.Tensor, None]] = None,
            box_coords: List[Union[List, torch.Tensor, None]] = None,
            return_all_hq_masks: bool = False,
            assemble_all_masks: bool = True
    ):
        if not hasattr(self, 'img_features'):
            raise RuntimeError(
                "Inference image has not been initialized! Please call set_infer_img() before infer().")
        if not hasattr(self, 'interm_features'):
            raise RuntimeError(
                "Inference image has not been initialized! Please call set_infer_img() before infer().")
        point_coords, point_labels, box_coords, noisy_masks = \
            self.proc_raw_prompts(point_coords=point_coords, box_coords=box_coords)

        masks_pred = self.forward(
            imgs=None, image_embeddings=self.img_features, interm_embeddings=self.interm_features,
            ori_img_size=self.ori_infer_img_size, return_all_hq_masks=return_all_hq_masks,
            point_coords=point_coords, point_labels=point_labels, box_coords=box_coords, noisy_masks=noisy_masks,
        )

        if assemble_all_masks:
            masks_pred = self.assemble_raw_masks(masks_pred)
        else:
            masks_pred = [self.discretize_mask(p_m) for p_m in masks_pred]
        return masks_pred


    def combine_all_box_masks(self, masks_all_boxes, box_num_list, return_logits: bool = False):
        masks_all_boxes = torch.stack(masks_all_boxes, dim=0)
        if not return_logits:
            masks_all_boxes = self.discretize_mask(masks_all_boxes)
        # assemble the masks predicted by all the boxes of a given image
        masks = []
        for i, box_num in enumerate(box_num_list):
            # for the image without boxes, we take the first mask prediction
            if box_num == 0:
                box_num = 1

            _masks = torch.sum(masks_all_boxes[:box_num, i], dim=0)
            masks.append(torch.clamp(_masks, max=1.0))

        masks = torch.stack(masks, dim=0)
        return masks


    @staticmethod
    def convert_raw_prompts_to_triple(
            point_coords: List[Union[torch.Tensor, None]],
            point_labels: List[Union[torch.Tensor, None]],
            box_coords: List[Union[torch.Tensor, None]],
            noisy_masks: List[Union[torch.Tensor, None]],
            batch_size: int
    ):
        points, boxes, masks = [], [], []
        for batch_idx in range(batch_size):
            points_idx = None
            if point_coords[batch_idx] is not None:
                # prompt point coordinates must in the shape of (N1, N2, 2)
                point_coords_idx = point_coords[batch_idx]
                if len(point_coords_idx.shape) == 2:
                    # if the first dim (N1) is not given, default to generate just one output mask
                    point_coords_idx = point_coords_idx.unsqueeze(0)
                if len(point_coords_idx.shape) != 3:
                    raise RuntimeError(
                        "Each prompt point coordinate in the list must be in the shape of (N1, N2, 2) "
                        "where N1 is the number of output masks and N2 is the number of prompt points!")
                if point_coords_idx.size(-1) != 2:
                    raise RuntimeError("Each prompt point must be given as a two-element vector!")

                point_labels_idx = point_labels[batch_idx]
                if len(point_labels_idx.shape) == 1:
                    # if the first dim (N1) is not given, default to generate just one output mask
                    point_labels_idx = point_labels_idx.unsqueeze(0)
                if len(point_labels_idx.shape) != 2:
                    raise RuntimeError(
                        "Each prompt point label in the list must be in the shape of (N1, N2) "
                        "where N1 is the number of output masks and N2 is the number of prompt points!")

                points_idx = (point_coords_idx, point_labels_idx)
            points.append(points_idx)

            boxes_idx = None
            if box_coords[batch_idx] is not None:
                # prompt box coordinates must in the shape of (N1, 2)
                box_coords_idx = box_coords[batch_idx]
                if len(box_coords_idx.shape) == 1:
                    # if the first dim (N1) is not given, default to generate just one output mask
                    box_coords_idx = box_coords_idx.unsqueeze(0)
                if len(box_coords_idx.shape) != 2:
                    raise RuntimeError(
                        "Each prompt point coordinate in the list must be in the shape of (N, 4) "
                        "where N is the number of output masks!")
                if box_coords_idx.size(-1) != 4:
                    raise RuntimeError("Each prompt box must be given as a four-element vector!")

                boxes_idx = box_coords_idx
            boxes.append(boxes_idx)

            masks_idx = None
            if noisy_masks[batch_idx] is not None:
                noisy_masks_idx = noisy_masks[batch_idx]
                if len(noisy_masks_idx.shape) == 2:
                    # if the first dim (N1) is not given, default to generate just one output mask
                    noisy_masks_idx = noisy_masks_idx[None, None, :, :]
                if len(noisy_masks_idx.shape) == 3:
                    noisy_masks_idx = noisy_masks_idx[None, :, :]
                if len(noisy_masks_idx.shape) != 4:
                    raise RuntimeError(
                        "Each prompt mask in the list must be in the shape of (N, 1, 256, 256) "
                        "where N is the number of output masks!")
                if noisy_masks_idx.size(1) != 1:
                    raise RuntimeError("Please only give one prompt mask for each output prompt!")
                if noisy_masks_idx.size(-2) != 256 and noisy_masks_idx.size(-1) != 256:
                    raise RuntimeError("Each prompt mask must have width and height of 256!")

                masks_idx = noisy_masks_idx
            masks.append(masks_idx)

        return points, boxes, masks


    def proc_raw_prompts(
            self,
            point_coords: List[Union[List, torch.Tensor, None]] = None,
            box_coords: List[Union[List, torch.Tensor, None]] = None,
            point_labels = None,
            noisy_masks = None,
    ):
        if not hasattr(self, 'ori_infer_img'):
            raise RuntimeError(
                "Image encoder features have not been registered! Please call set_infer_img() before infer().")
        if not hasattr(self, 'ori_infer_img_size'):
            raise RuntimeError(
                "Image encoder features have not been registered! Please call set_infer_img() before infer().")

        if isinstance(point_coords, torch.Tensor):
            if len(point_coords.shape) == 2:
                point_coords = [point_coords.unsqueeze(0)]
            elif len(point_coords.shape) == 3:
                point_coords = [point_coords]
            elif len(point_coords.shape) == 4:
                point_coords = [p_c for p_c in point_coords]
            else:
                raise RuntimeError
        elif not isinstance(point_coords, (List, type(None))):
            raise RuntimeError

        if isinstance(box_coords, torch.Tensor):
            if len(box_coords.shape) == 1:
                box_coords = [box_coords.unsqueeze(0)]
            elif len(box_coords.shape) <= 2:
                box_coords = [box_coords]
            elif len(box_coords.shape) == 3:
                box_coords = [b_c for b_c in box_coords]
            else:
                raise RuntimeError
        elif not isinstance(box_coords, (List, type(None))):
            raise RuntimeError

        def proc_coords(input_coords):
            if input_coords is not None:
                _tmp_input_coords = []
                for i_c in input_coords:
                    if i_c is None or len(i_c[0]) == 0:
                        _tmp_input_coords.append(None)
                    elif isinstance(i_c, torch.Tensor):
                        _tmp_input_coords.append(i_c)
                    elif len(i_c[0]) > 0:
                        _tmp_input_coords.append(torch.FloatTensor(i_c).cuda())
                    else:
                        raise RuntimeError(
                            "The element of the input coords must be one of List, torch.Tensor, and None. "
                            f"Got {type(i_c)}!!"
                        )
                input_coords = _tmp_input_coords
            return input_coords

        # produce prompt points and their corresponding labels
        point_coords, point_labels = proc_coords(point_coords), proc_coords(point_labels)
        if point_coords is None:
            point_coords = [None for _ in self.ori_infer_img]
            point_labels = [None for _ in self.ori_infer_img]
        elif point_labels is None:
            point_labels = [
                torch.ones((p_c.size(0), p_c.size(1)), dtype=torch.long, device=p_c.device) if p_c is not None else None
                for p_c in point_coords
            ]

        # produce prompt boxes
        box_coords = proc_coords(box_coords)
        if box_coords is None:
            box_coords = [None for _ in self.ori_infer_img]

        # preprocess the value scale of points and boxes
        _, point_coords, box_coords = self.preprocess(
            point_coords=point_coords, box_coords=box_coords, ori_img_size=self.ori_infer_img_size
        )

        # We disable prompt masks for prediction
        if noisy_masks is None:
            noisy_masks = [None for _ in self.ori_infer_img]

        return point_coords, point_labels, box_coords, noisy_masks


    def assemble_raw_masks(self, raw_masks: List):
        # Order: discretize -> sum over all output masks -> clamp the values larger than 1 -> stack into a batch
        masks = []
        for r_m in raw_masks:
            # discretize the logits into a 0-1 mask
            r_m = self.discretize_mask(r_m)
            # sum up the prediced masks by all the prompts of a single image
            r_m = torch.sum(r_m, dim=0, keepdim=True)
            masks.append(torch.clamp(r_m, max=1.0))
        return masks


    def discretize_mask(self, masks_logits):
        return torch.gt(masks_logits, self.ori_sam.mask_threshold).float()


    def preprocess(self,
                   ori_img_size: List[Tuple],
                   imgs: List[torch.Tensor] = None,
                   point_coords: List[Union[torch.Tensor, None]] = None,
                   box_coords: List[Union[torch.Tensor, None]] = None):
        # rescale the data in the out-place manner
        imgs_return, point_coords_return, box_coords_return = \
            copy.deepcopy(imgs), copy.deepcopy(point_coords), copy.deepcopy(box_coords)
        # resize each image into a 4-dim tensor for interpolate
        if imgs_return is not None:
            for i in range(len(imgs_return)):
                if len(imgs_return[i].shape) == 3:
                    imgs_return[i] = imgs_return[i].unsqueeze(0)
                if len(imgs_return[i].shape) != 4:
                    raise RuntimeError(
                        f'Wrong image shape! Each image in your given list must be (C, H, W), '
                        f'but got {imgs_return[i].shape}!'
                    )

        # loop each image
        for i in range(len(ori_img_size)):
            # skip the one with the same size as SAM input
            if ori_img_size[i] == self.sam_img_size:
                continue

            if imgs_return is not None:
                # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
                # change the mode from bilinear to nearest
                imgs_return[i] = F.interpolate(
                    imgs_return[i], self.sam_img_size, mode="bilinear", align_corners=False,
                )
                # Normalize colors to match the original SAM preprocessing
                imgs_return[i] = (imgs_return[i] - self.ori_sam.pixel_mean) / self.ori_sam.pixel_std

            h_scale = self.sam_img_size[0] / ori_img_size[i][0]
            w_scale = self.sam_img_size[1] / ori_img_size[i][1]
            if point_coords_return is not None and point_coords_return[i] is not None:
                # scale the x-axis by the width scaler
                point_coords_return[i][:, :, 0] *= w_scale
                # scale the y-axis by the height scaler
                point_coords_return[i][:, :, 1] *= h_scale
                # make sure that the point coordinates are in the form of integers
                point_coords_return[i] = torch.round(point_coords_return[i])
            if box_coords_return is not None and box_coords_return[i] is not None:
                # scale the x-axis by the width scaler
                box_coords_return[i][:, 0] *= w_scale
                box_coords_return[i][:, 2] *= w_scale
                # scale the y-axis by the height scaler
                box_coords_return[i][:, 1] *= h_scale
                box_coords_return[i][:, 3] *= h_scale
                # make sure that the point coordinates are in the form of integers
                box_coords_return[i] = torch.round(box_coords_return[i])

        # organize all the image tensors into a larger matrix
        if imgs_return is not None:
            imgs_return = torch.cat(imgs_return, dim=0)
        return imgs_return, point_coords_return, box_coords_return


    @staticmethod
    def postprocess(output_masks: torch.Tensor, ori_img_size: Tuple):
        # rescale the mask size back to original image size
        output_mask_size = (output_masks.size(-2), output_masks.size(-1))
        if output_mask_size != ori_img_size:
            if len(output_masks.shape) == 3:
                output_masks = output_masks.unsqueeze(1)
            # bilinear will produce non-deterministic gradients during training. For exact reproduction, please
            # change the mode from bilinear to nearest
            output_masks = F.interpolate(
                output_masks, ori_img_size, mode="bilinear", align_corners=False,
            )
        return output_masks



class CATSAMT(BaseCATSAM):
    def __init__(self, model_type: str):
        super(CATSAMT, self).__init__(model_type=model_type)
        self.image_encoder = CATSAMTImageEncoder(ori_sam=self.ori_sam, hq_token=self.mask_decoder.hf_token.weight)


class CATSAMA(BaseCATSAM):
    def __init__(self, model_type: str):
        super(CATSAMA, self).__init__(model_type=model_type)
        self.image_encoder = CATSAMAImageEncoder(ori_sam=self.ori_sam, hq_token=self.mask_decoder.hf_token.weight)
