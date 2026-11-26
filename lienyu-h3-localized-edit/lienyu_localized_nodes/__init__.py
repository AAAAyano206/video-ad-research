"""Lock H3's packed audio/video state to the original shot after each Turbo step."""
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from comfy.samplers import KSAMPLER
from safetensors.torch import load_file, save_file


def lock_weights(pixel_masks, mask_shape, attribute):
    # The deployed video VAE decodes five time tokens into 17 frames.
    # Each decoder block discards its first three padding frames.
    if (mask_shape[2] - 2) % 5:
        raise ValueError(f"unrecognized H3 VAE time dimension: {mask_shape[2]}")
    windows = []
    last = len(pixel_masks) - 1
    for token in range(mask_shape[2]):
        chunk, index = divmod(token, 5)
        start = min(last, max(0, 17 * chunk + 4 * index - 5))
        stop = min(last + 1, 17 * chunk + 4 * (index + 1) - 3 + 2)
        windows.append(pixel_masks[start:max(start + 1, stop)].amax(0))
    masks = torch.stack(windows).float() / 255
    masks = F.adaptive_max_pool2d(masks.unsqueeze(1), mask_shape[-2:])
    radius = 1 if attribute == "paint" else 2
    masks = F.max_pool2d(masks, 2 * radius + 1, stride=1, padding=radius)
    return masks[:, 0].unsqueeze(0).unsqueeze(0).expand(mask_shape).reshape(mask_shape[0], -1)


class LocalizedH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sampler": ("SAMPLER",),
            "latent": ("LATENT",),
            "case_id": ("STRING",),
            "mode": (["record", "lock"],),
            "attribute": (["paint", "body"],),
            "mask_sha256": ("STRING",),
            "mask_directory": ("STRING",),
            "trajectory_sha256": ("STRING",),
        }}

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build"
    CATEGORY = "lienyu/H3"

    def build(self, sampler, latent, case_id, mode, attribute, mask_sha256, mask_directory, trajectory_sha256):
        if not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
            raise ValueError("case_id must be a filename")
        if mask_directory not in {"masks", "masks-segmented"}:
            raise ValueError("unexpected mask directory")
        video, audio = latent["samples"].unbind()
        if video.ndim != 5 or audio.ndim != 4:
            raise ValueError("expected H3 video and audio latents")
        root = Path(os.environ["LIENYU_LOCALIZED_RUN_ROOT"])
        trajectory = root / "trajectories" / (case_id + ".safetensors")
        full_mask = None
        anchors = None
        if mode == "lock":
            masks = load_file(str(root / mask_directory / (case_id + ".safetensors")))["pixel_mask"]
            video_mask = lock_weights(masks, video.shape, attribute)
            audio_mask = torch.zeros((audio.shape[0], audio[0].numel()))
            full_mask = torch.cat((video_mask, audio_mask), dim=-1).unsqueeze(1)
            anchors = load_file(str(trajectory))

        def sample_with_constraint(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            if mode == "lock":
                expected = anchors["sigmas"].to(sigmas.device)
                if not sigmas.equal(expected) or not x.equal(anchors["initial"].to(x.device)):
                    raise ValueError("H3 schedule or actual initial noise differs from the A0 trajectory")
                if full_mask.shape != x.shape:
                    raise ValueError(f"packed mask shape {full_mask.shape} differs from latent {x.shape}")
                editable = full_mask.to(device=x.device, dtype=torch.bool)
                locked_steps = [anchors[f"step_{i}"].to(x.device) for i in range(len(sigmas) - 1)]
            else:
                snapshots = {"sigmas": sigmas.detach().cpu().clone(), "initial": x.detach().cpu().clone()}

            def after_step(values):
                state = values["x"]
                step = values["i"]
                if mode == "lock":
                    state.copy_(torch.where(editable, state, locked_steps[step]))
                else:
                    snapshots[f"step_{step}"] = state.detach().cpu().clone()
                if callback:
                    callback(values)

            samples = sampler.sampler_function(
                model, x, sigmas, extra_args=extra_args, callback=after_step, disable=disable, **kwargs
            )
            if mode == "record":
                temporary = trajectory.with_suffix(".tmp")
                save_file(snapshots, str(temporary))
                temporary.replace(trajectory)
            return samples

        return (KSAMPLER(sample_with_constraint, sampler.extra_options, sampler.inpaint_options),)


NODE_CLASS_MAPPINGS = {"LienyuLocalizedH3Sampler": LocalizedH3Sampler}
