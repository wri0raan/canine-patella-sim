import deeplabcut
CASES = [("dlcrnet", "superanimal_quadruped", "dlcrnet", None), ("fulldog_hrnet", "full_dog", "hrnet_w32", "fasterrcnn_resnet50_fpn_v2"), ("fulldog_resnet", "full_dog", "resnet_50", "fasterrcnn_resnet50_fpn_v2")]
for tag, sa, model, det in CASES:
    for c in ("ca", "cb"):
        try:
            kw = dict(detector_name=det, detector_batch_size=4, bbox_threshold=0.5) if det else {}
            deeplabcut.video_inference_superanimal([f"data/real/leo5/work/{c}.mp4"], superanimal_name=sa, model_name=model, dest_folder=f"data/real/leo5/work/{tag}", max_individuals=1, pcutoff=0.1, batch_size=8, create_labeled_video=False, **kw)
            print("done", tag, c, flush=True)
        except Exception as e:
            print("FAILED", tag, c, repr(e)[:400], flush=True)
