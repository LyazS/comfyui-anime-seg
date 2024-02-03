import os
import huggingface_hub
import torch
import onnxruntime as rt
import numpy as np
import cv2


# Declare Execution Providers
ort_providers = {
    "GPU": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "CPU": ["CPUExecutionProvider"],
}

# init the model
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
if not os.path.exists(current_dir):
    os.makedirs(current_dir, exist_ok=True)
model_path = os.path.join(current_dir, "isnetis.onnx")
if not os.path.exists(model_path):
    huggingface_hub.hf_hub_download(
        "skytnt/anime-seg",
        local_dir=model_path,
    )
rmbg_model = rt.InferenceSession(model_path, providers=ort_providers["CPU"])


def get_mask(img: torch.Tensor, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {"img": img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask


def rmbg_fn(img):
    mask = get_mask(img)
    mask = (mask * 255).astype(np.uint8)
    return mask


class AnimeCharacterSEG:
    def __init__(self):
        self.device = "CPU"
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "device": (["CPU", "GPU"], {"default": "CPU"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "abg_remover"
    CATEGORY = "mask"

    def abg_remover(self, images: torch.Tensor, device: str):
        global rmbg_model
        if device != self.device:
            self.device = device
            rmbg_model = rt.InferenceSession(
                model_path, providers=ort_providers[self.device]
            )
            pass
        batch_tensor = []
        for image in images:
            npa = image2nparray(image)
            mask = rmbg_fn(npa)
            mask = nparray2image(mask)
            batch_tensor.append(mask)

        batch_tensor = torch.cat(batch_tensor, dim=0)
        return (batch_tensor,)


def image2nparray(image: torch.Tensor):
    narray: np.array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    # cvt color
    if narray.shape[-1] == 4:
        narray = narray[..., [2, 1, 0, 3]]  # For RGBA
    else:
        narray = narray[..., [2, 1, 0]]  # For RGB
    return narray


def nparray2image(narray: np.array):
    narray = np.transpose(narray, (2, 0, 1))
    tensor = torch.from_numpy(narray / 255.0).float().unsqueeze(0)
    return tensor


NODE_CLASS_MAPPINGS = {"Anime Character Seg": AnimeCharacterSEG}
