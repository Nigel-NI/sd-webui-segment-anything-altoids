import copy
from typing import Tuple, List, Dict
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from modules import shared


def max_cn_num():
    if shared.opts.data is None:
        return 1
    if 'control_net_unit_count' in shared.opts.data:
        return int(shared.opts.data.get('control_net_unit_count', 1))
    return int(shared.opts.data.get('control_net_max_models_num', 1))


def open_gallery_image(item):
    if item is None:
        return None
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, str):
        return Image.open(item)
    if isinstance(item, dict):
        image_path = item.get('name') or item.get('path')
        if image_path is not None:
            return Image.open(image_path)
    if isinstance(item, (tuple, list)) and len(item) > 0:
        return open_gallery_image(item[0])
    raise TypeError(f"Cannot locate image in gallery item of type {type(item)}")


def pil_to_controlnet_array(image, mode="RGB"):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image.astype("uint8", copy=False)
    if isinstance(image, Image.Image):
        return np.array(image.convert(mode)).astype("uint8", copy=False)
    return np.array(image).astype("uint8", copy=False)


def get_controlnet_script(p):
    script_runner = getattr(p, 'scripts', None)
    for script in getattr(script_runner, 'alwayson_scripts', []):
        try:
            if script.title() == "ControlNet":
                return script
        except Exception:
            continue
    return None


def update_forge_controlnet_unit(p, idx: int, image, mask=None, module="None"):
    if idx is None:
        return False
    idx = int(idx)
    controlnet_script = get_controlnet_script(p)
    script_args = getattr(p, 'script_args', None)
    if controlnet_script is None or script_args is None:
        return False

    args_from = getattr(controlnet_script, 'args_from', None)
    args_to = getattr(controlnet_script, 'args_to', None)
    if args_from is None or args_to is None or idx < 0 or idx >= (args_to - args_from):
        return False

    arg_index = args_from + idx
    if arg_index >= len(script_args):
        return False

    try:
        from lib_controlnet.external_code import ControlNetUnit
    except Exception:
        ControlNetUnit = None

    unit = script_args[arg_index]
    if ControlNetUnit is not None and unit is None:
        unit = ControlNetUnit(enabled=False, module="None", model="None")
    elif isinstance(unit, dict):
        unit = unit.copy()
    else:
        unit = copy.copy(unit)

    image_array = pil_to_controlnet_array(image, "RGB")
    mask_array = pil_to_controlnet_array(mask, "L") if mask is not None else None

    if isinstance(unit, dict):
        unit['image'] = image_array
        unit['module'] = module
        if mask_array is not None:
            unit['mask_image'] = mask_array
    elif unit is not None and hasattr(unit, 'image'):
        unit.image = image_array
        if hasattr(unit, 'module'):
            unit.module = module
        if mask_array is not None and hasattr(unit, 'mask_image'):
            unit.mask_image = mask_array
    else:
        return False

    updated_args = list(script_args)
    updated_args[arg_index] = unit
    p.script_args = tuple(updated_args) if isinstance(script_args, tuple) else updated_args
    return True


class SAMInpaintUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img

        self.inpaint_upload_enable: bool = False
        self.cnet_inpaint_invert: bool = False
        self.cnet_inpaint_idx: int = 0
        self.input_image = None
        self.output_mask_gallery: List[Dict] = None
        self.output_chosen_mask: int = 0
        self.dilation_checkbox: bool = False
        self.dilation_output_gallery: List[Dict] = None
        self.init_sam_single_image_process(args)

    
    def init_sam_single_image_process(self, args):
        self.inpaint_upload_enable      = args[0]
        self.cnet_inpaint_invert        = args[1]
        self.cnet_inpaint_idx           = args[2]
        self.input_image                = args[3]
        self.output_mask_gallery        = args[4]
        self.output_chosen_mask         = args[5]
        self.dilation_checkbox          = args[6]
        self.dilation_output_gallery    = args[7]


    def get_input_and_mask(self, mask_blur):
        image, mask = None, None
        if self.inpaint_upload_enable and self.input_image is not None and self.output_mask_gallery is not None:
            if self.dilation_checkbox and self.dilation_output_gallery is not None:
                mask = open_gallery_image(self.dilation_output_gallery[1]).convert('L')
            elif self.output_mask_gallery is not None:
                mask = open_gallery_image(self.output_mask_gallery[self.output_chosen_mask + 3]).convert('L')
            if mask is not None and self.cnet_inpaint_invert:
                mask = ImageOps.invert(mask)
            # if self.is_img2img and self.sketch_checkbox and self.inpaint_color_sketch is not None and mask is not None:
            #     alpha = np.expand_dims(np.array(mask) / 255, axis=-1)
            #     image = np.uint8(np.array(self.inpaint_color_sketch) * alpha + np.array(self.input_image) * (1 - alpha))
            #     mask = ImageEnhance.Brightness(mask).enhance(1 - self.inpaint_mask_alpha / 100)
            #     blur = ImageFilter.GaussianBlur(mask_blur)
            #     image = Image.composite(image.filter(blur), self.input_image, mask.filter(blur)).convert("RGB")
            # else:
            image = self.input_image
        return image, mask



class SAMProcessUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img
        self.sam_inpaint_unit = SAMInpaintUnit(args, is_img2img)

        args = args[8:]
        self.cnet_seg_output_gallery: List[Dict]    = None
        self.cnet_seg_enable_copy: bool             = False
        self.cnet_seg_idx: int                      = 0
        self.cnet_seg_gallery_input: int            = 0
        self.init_cnet_seg_process(args)

        args = args[4:]
        self.crop_inpaint_unit = SAMInpaintUnit(args, is_img2img)

        args = args[8:]
        self.cnet_upload_enable: bool                   = False
        self.cnet_upload_num: int                       = 0
        self.cnet_upload_img_inpaint: Image.Image       = None
        self.cnet_upload_mask_inpaint: Image.Image      = None
        self.init_cnet_upload_process(args)
        
        
    def init_cnet_seg_process(self, args):
        self.cnet_seg_output_gallery    = args[0]
        self.cnet_seg_enable_copy       = args[1]
        self.cnet_seg_idx               = args[2]
        self.cnet_seg_gallery_input     = args[3]
    

    def init_cnet_upload_process(self, args):
        self.cnet_upload_enable         = args[0]
        self.cnet_upload_num            = args[1]
        self.cnet_upload_img_inpaint    = args[2] 
        self.cnet_upload_mask_inpaint   = args[3]

    
    def set_process_attributes(self, p):
        inpaint_mask_blur = getattr(p, "mask_blur", 0)
        inpaint_image, inpaint_mask = self.sam_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
        inpaint_cn_num = self.sam_inpaint_unit.cnet_inpaint_idx
        if inpaint_image is None:
            inpaint_image, inpaint_mask = self.crop_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
            inpaint_cn_num = self.crop_inpaint_unit.cnet_inpaint_idx
        if inpaint_image is not None and inpaint_mask is not None:
            if self.is_img2img:
                p.init_images = [inpaint_image]
                p.image_mask = inpaint_mask
            else:
                update_forge_controlnet_unit(p, inpaint_cn_num, inpaint_image, inpaint_mask)
                self.set_p_value(p, 'control_net_input_image', inpaint_cn_num, 
                                {"image": inpaint_image, "mask": inpaint_mask.convert("L")})
        
        if self.cnet_seg_enable_copy and self.cnet_seg_output_gallery is not None:
            cnet_seg_gallery_index = 1
            if len(self.cnet_seg_output_gallery) == 3 and self.cnet_seg_gallery_input is not None:
                cnet_seg_gallery_index += self.cnet_seg_gallery_input
            cnet_seg_image = open_gallery_image(self.cnet_seg_output_gallery[cnet_seg_gallery_index])
            update_forge_controlnet_unit(p, self.cnet_seg_idx, cnet_seg_image)
            self.set_p_value(p, 'control_net_input_image', self.cnet_seg_idx, 
                             cnet_seg_image)
        
        if self.cnet_upload_enable and self.cnet_upload_img_inpaint is not None and self.cnet_upload_mask_inpaint is not None:
            update_forge_controlnet_unit(p, self.cnet_upload_num, self.cnet_upload_img_inpaint, self.cnet_upload_mask_inpaint)
            self.set_p_value(p, 'control_net_input_image', self.cnet_upload_num, 
                            {"image": self.cnet_upload_img_inpaint, "mask": self.cnet_upload_mask_inpaint.convert("L")})


    def set_p_value(self, p, attr: str, idx: int, v):
        if idx is None:
            return
        idx = int(idx)
        value = getattr(p, attr, None)
        if isinstance(value, list):
            while len(value) <= idx:
                value.append(None)
            value[idx] = v
        else:
            # if value is None, ControlNet uses default value
            value = [value] * max(max_cn_num(), idx + 1)
            value[idx] = v
        setattr(p, attr, value)
