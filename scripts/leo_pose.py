import sys
import deeplabcut

video, out = sys.argv[1], sys.argv[2]
labeled = "--labeled" in sys.argv
deeplabcut.video_inference_superanimal(
    [video], superanimal_name="superanimal_quadruped", model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2", dest_folder=out, max_individuals=1, pcutoff=0.1,
    batch_size=8, detector_batch_size=4, create_labeled_video=labeled, plot_bboxes=labeled, bbox_threshold=0.5)
print("done")
