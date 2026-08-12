import torch
import torch.nn.functional as F


class ConvertMCSkinXZ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_skin"
    CATEGORY = "xzuynodes"
    DESCRIPTION = "Converts a 2:1 side-by-side skin image (left RGB, right alpha mask) into a 64x64 RGBA Minecraft skin."

    def convert_skin(self, image):
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

        rgba_64 = torch.cat([skin_64, mask_64], dim=1)
        out_image = rgba_64.permute(0, 2, 3, 1)

        return (out_image,)


NODE_CLASS_MAPPINGS = {"ConvertMCSkinXZ": ConvertMCSkinXZ}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertMCSkinXZ": "Convert MC Skin (XZ)"}
