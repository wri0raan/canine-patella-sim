import sys, deeplabcut
for i in range(1,5):
    deeplabcut.video_inference_superanimal([f"data/real/leo4/n{i}.mp4"], superanimal_name="superanimal_quadruped", model_name="hrnet_w32", detector_name="fasterrcnn_resnet50_fpn_v2", dest_folder="data/real/leo4/work", max_individuals=1, pcutoff=0.1, batch_size=8, detector_batch_size=4, create_labeled_video=False, bbox_threshold=0.5)
    print("done", i, flush=True)
