import torch
import torch.nn.functional as F


class ConvertMCSkinXZ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "safe_alpha": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Treat alpha mask values below safe_alpha_min as 0.0 (fully transparent), and values above safe_alpha_max as 1.0 (fully opaque).",
                    },
                ),
            },
            "optional": {
                "safe_alpha_min": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold for 0.0 opacity.",
                    },
                ),
                "safe_alpha_max": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Threshold for 1.0 opacity.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_skin"
    CATEGORY = "xzuynodes"
    DESCRIPTION = "Converts a 2:1 side-by-side skin image (left RGB, right alpha mask) into a 64x64 RGBA Minecraft skin."

    def convert_skin(self, image, safe_alpha, safe_alpha_min=0.01, safe_alpha_max=0.99):
        _, H, W, _ = image.shape
        half_w = W // 2

        left_rgb = image[:, :, :half_w, :3]
        right_mask = image[:, :, half_w:, :3]

        luma_weights = torch.tensor(
            [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
        )
        mask_gray = torch.sum(right_mask * luma_weights, dim=-1, keepdim=True)

        left_rgb_p = left_rgb.permute(0, 3, 1, 2)
        mask_gray_p = mask_gray.permute(0, 3, 1, 2)

        skin_64 = F.interpolate(left_rgb_p, size=(64, 64), mode="nearest-exact")
        mask_64 = F.interpolate(mask_gray_p, size=(64, 64), mode="nearest-exact")

        if safe_alpha:
            clean_alpha = torch.where(
                mask_64 < safe_alpha_min,
                0.0,
                torch.where(mask_64 > safe_alpha_max, 1.0, mask_64),
            )
        else:
            clean_alpha = mask_64

        rgba_64 = torch.cat([skin_64, clean_alpha], dim=1)
        out_image = rgba_64.permute(0, 2, 3, 1)

        return (out_image,)


NODE_CLASS_MAPPINGS = {"ConvertMCSkinXZ": ConvertMCSkinXZ}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertMCSkinXZ": "Convert MC Skin (XZ)"}
