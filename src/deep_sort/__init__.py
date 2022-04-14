from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker():
    return DeepSort("/home/nehalnevle/internship_int2peta/image_pose_ws/src/yolov5_pytorch_ros/src/deep_sort/deep/checkpoint/ckpt.t7", 
                max_dist=0.2, min_confidence=0.17, 
                nms_max_overlap=0.5, max_iou_distance=0.7, 
                max_age=70, n_init=3, nn_budget=100, use_cuda=False)
    









