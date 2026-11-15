import sys, deeplabcut
for tag in sys.argv[1:]:
    for model in ("hrnet_w32", "resnet_50"):
        deeplabcut.video_inference_superanimal([f"data/real/leo5/work/c{tag}.mp4"], superanimal_name="superanimal_quadruped", model_name=model, detector_name="fasterrcnn_resnet50_fpn_v2", dest_folder=f"data/real/leo5/work/{model}",
                                               max_individuals=1, pcutoff=0.1, batch_size=8, detector_batch_size=4, create_labeled_video=False, bbox_threshold=0.5)
        print("done", model, tag, flush=True)
