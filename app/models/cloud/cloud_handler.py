"""Hosted image models (fal.ai, Replicate).

These models have no local weights: the controller sends the prompt to the
provider API (services.cloud_image) instead of loading a pipeline.
cloud_flux_lora models accept FLUX.1 dev LoRAs and share the local flux LoRA
folder, so CivitAI downloads for Flux.1 D show up for them too.
"""
import os


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        cloud = model_def.get("cloud") or {}
        extra_model_def = {
            "image_outputs": True,
            "guidance_max_phases": 1 if cloud.get("guidance_field") else 0,
            "loras_disabled": not cloud.get("lora"),
            "profiles_dir": [],
            "no_background_removal": True,
            "URLs": [],
        }
        if cloud.get("image_field"):
            extra_model_def["image_ref_choices"] = {
                "choices": [("None", ""), ("Reference / source images", "I")],
                "letters_filter": "I",
                "default": "I" if cloud.get("requires_image") else "",
            }
            extra_model_def["one_image_ref_needed"] = bool(cloud.get("requires_image"))
        return extra_model_def

    @staticmethod
    def query_supported_types():
        return ["cloud_image", "cloud_flux_lora"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "cloud"

    @staticmethod
    def query_family_infos():
        return {"cloud": (190, "Cloud API")}

    @staticmethod
    def get_lora_dir(base_model_type, args, lora_root):
        if base_model_type == "cloud_flux_lora":
            return getattr(args, "lora_dir_flux", None) or os.path.join(lora_root, "flux")
        return os.path.join(lora_root, "cloud")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return []

    @staticmethod
    def load_model(*args, **kwargs):
        raise RuntimeError("Cloud models run through their provider API, not a local pipeline")

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "num_inference_steps": 1,
            "guidance_scale": 0,
            "batch_size": 1,
            **((model_def.get("cloud") or {}).get("defaults") or {}),
        })
