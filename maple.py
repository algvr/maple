from PIL import Image
import torch
from torchvision import transforms


class MAPLE(torch.nn.Module):
    def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):
        super(MAPLE, self).__init__()
        self.device = device
        self.dtype = dtype
        self.p = 16
        self.stride = 16
        self.feats = []
        self.extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(self.device, dtype=self.dtype)

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self.feats.append(qkv[1])

        # extract features from layer 11 of the ViT
        self.hook_handler = self.extractor.blocks[11].attn.register_forward_hook(_inner_hook)

        class SquarePad:
            # from https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
            def __call__(self, image):
                image_size = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
                max_wh = max(image_size)
                p_left, p_top = [(max_wh - s) // 2 for s in image_size]
                p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image_size, [p_left, p_top])]
                padding = (0, 0, p_left + p_right, p_top + p_bottom)
                return transforms.functional.pad(image, padding, 0, "constant")

        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict["model"] = {k: v.to(dtype=self.dtype) for k, v in state_dict["model"].items()}
        encoder_keys = {k.replace("model.", "")[len("encoder."):]: v for k, v in state_dict["model"].items() if k.startswith("encoder.")}
        self.load_state_dict(encoder_keys)

    def forward_model(self, batch):
        B, C, H, W = batch.shape
        self.feats.clear()
        _ = self.extractor(batch)
        x = self.feats[0]
        desc = x.permute(0, 2, 3, 1)[..., 0, :].flatten(start_dim=1, end_dim=-1)  # Bxd
        return desc

    def forward(self, input_imgs):
        if not isinstance(input_imgs, list):
            input_imgs = [input_imgs]

        inputs = torch.stack([self.transform(input_img).to(self.device, dtype=self.dtype) for input_img in input_imgs], dim=0)
        desc = self.forward_model(inputs)
        if len(desc.shape) == 1:
            desc = desc[None, ...]
        return desc
