import os
import folder_paths
import numpy as np
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar
import cv2
from .trt_utilities import Engine
from torchvision.transforms.functional import normalize

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


ENGINE_DIR = os.path.join(folder_paths.models_dir,"tensorrt", "facerestore")

class FaceRestoreTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, engine):

        # setup tensorrt engine
        engine = Engine(os.path.join(ENGINE_DIR,engine))
        engine.load()
        engine.activate()
        engine.allocate_buffers()

        cudaStream = torch.cuda.current_stream().cuda_stream
        pbar = ProgressBar(images.shape[0])
        images = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images, size=(512,512), mode='bilinear', align_corners=False)
        images_list = list(torch.split(images_resized, split_size_or_sections=1))

        output_frames = []

        for img in images_list:
            normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            result = engine.infer({"input": img},cudaStream)
            output = result['output']

            output = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            output = output.astype('uint8')
            output = cv2.resize(output, (images.shape[3], images.shape[2]))
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            output_frames.append(output)
            pbar.update(1)


        output_frames = np.array(output_frames).astype(np.float32) / 255.0
        return (torch.from_numpy(output_frames),)
        
NODE_CLASS_MAPPINGS = { 
    "FaceRestoreTensorrt" : FaceRestoreTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
     "FaceRestoreTensorrt" : "Face Restore Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']