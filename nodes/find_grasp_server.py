#! /home/rdml/miniconda3/envs/tsgrasp/bin/python
# shebang is for the Python3 environment with all dependencies

# tsgrasp dependencies
import sys
from typing import List
sys.path.append("/home/rdml/Research/grasp_synth_ws/src/grasp_synth/")
from nn.load_model import load_model

# ROS dependencies
import rospy
from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import ros_numpy
from grasp_synth.msg import Grasp, Grasps
from grasp_synth.srv import FindGrasps, FindGraspsRequest, FindGraspsResponse

# python dependencies
import numpy as np

import torch
from threading import Lock
from kornia.geometry.conversions import rotation_matrix_to_quaternion, QuaternionCoeffOrder
import MinkowskiEngine as ME
# from pytorch3d.ops import sample_farthest_points
torch.backends.cudnn.benchmark=True # makes a big difference on FPS for some PTS_PER_FRAME values, but seems to increase memory usage and can result in OOM errors.

## global constants
PTS_PER_FRAME   = 45000
GRIPPER_DEPTH   = 0.1034
CONF_THRESHOLD  = 0
TOP_K           = 400

## load Pytorch Lightning network
device = torch.device('cuda')
pl_model = load_model().to(device)
pl_model.eval()

gpu_mtx = Lock() # mutex so we only process one point cloud at a time
latest_header = Header()

ee_pose_msg = None

def ee_pose_cb(msg):
    global ee_pose_msg
    ee_pose_msg = msg

import time
class TimeIt:
    """ Class for timing code. Used as context decorator."""
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = True

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        # torch.cuda.synchronize()
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))

#@torch.jit.script
def build_6dof_grasps(contact_pts, baseline_dir, approach_dir, grasp_width, gripper_depth: float=GRIPPER_DEPTH):
    """Calculate the SE(3) transforms corresponding to each predicted coord/approach/baseline/grasp_width grasp.

    Unbatched for torch.jit.script.

    Args:
        contact_pts (torch.Tensor): (N, 3) contact points predicted
        baseline_dir (torch.Tensor): (N, 3) gripper baseline directions
        approach_dir (torch.Tensor): (N, 3) gripper approach directions
        grasp_width (torch.Tensor): (N, 3) gripper width

    Returns:
        pred_grasp_tfs (torch.Tensor): (N, 4, 4) homogeneous grasp poses.
    """
    N = contact_pts.shape[0]
    grasps_R = torch.stack([baseline_dir, torch.cross(approach_dir, baseline_dir), approach_dir], dim=-1)
    grasps_t = contact_pts + grasp_width/2 * baseline_dir - gripper_depth * approach_dir
    ones = torch.ones((N, 1, 1), device=contact_pts.device)
    zeros = torch.zeros((N, 1, 3), device=contact_pts.device)
    homog_vec = torch.cat([zeros, ones], dim=-1)

    pred_grasp_tfs = torch.cat([
        torch.cat([grasps_R, grasps_t.unsqueeze(-1)], dim=-1), 
        homog_vec
    ], dim=-2)
    return pred_grasp_tfs

#@torch.jit.script
def discretize(positions: torch.Tensor, grid_size: float) -> torch.Tensor:
    """Truncate each position to an integer grid."""
    return (positions / grid_size).int()

#@torch.jit.script
def prepend_coordinate(matrix: torch.Tensor, coord: int):
        """Concatenate a constant column of value `coord` before a 2D matrix."""
        return torch.column_stack([
            coord * torch.ones((len(matrix), 1), device=matrix.device),
            matrix
        ])

def unweighted_sum(coords: torch.Tensor):
    """Create a feature vector from a coordinate array, so each 
    row's feature is the number of rows that share that coordinate."""
    
    unique_coords, idcs, counts = coords.unique(dim=0, return_counts=True, return_inverse=True)
    features = counts[idcs]
    return features.reshape(-1, 1).to(torch.float32)

def infer_grasps(tsgraspnet, points: List[torch.Tensor], grid_size: float) -> torch.Tensor:
    """Run a sparse convolutional network on a list of consecutive point clouds, and return the grasp predictions for the last point cloud. Each point cloud may have different numbers of points."""

    ## Convert list of point clouds into matrix of 4D coordinate
    coords = list(map(
        lambda mtx_coo: prepend_coordinate(*mtx_coo),
        zip(points, range(len(points)))
    ))
    coords = torch.cat(coords, dim=0)
    coords = prepend_coordinate(coords, 0) # add dummy batch dimension

    ## Discretize coordinates to integer grid
    coords = discretize(coords, grid_size).contiguous()
    feats = unweighted_sum(coords)

    ## Construct a Minkoswki sparse tensor and run forward inference
    stensor = ME.SparseTensor(
        coordinates = coords,
        features = feats
    )

    with TimeIt("   tsgraspnet.model.forward"):
        class_logits, baseline_dir, approach_dir, grasp_offset = tsgraspnet.model.forward(stensor)

    ## Return the grasp predictions for the latest point cloud
    idcs = coords[:,1] == coords[:,1].max()
    return(
        class_logits[idcs], baseline_dir[idcs], approach_dir[idcs], grasp_offset[idcs], points[-1]
    )

#@torch.jit.script
def downsample_xyz(pts: List[torch.Tensor], pts_per_frame: int) -> List[torch.Tensor]:
    ## downsample point clouds proportion of points -- will that result in same sampling distribution?
    for i in range(len(pts)):
        pts_to_keep = int(pts_per_frame / 90_000 * len(pts[i]))
        idxs = torch.randperm(
            len(pts[i]), dtype=torch.int32, device=pts[i].device
        )[:pts_to_keep].sort()[0].long()

        pts[i] = pts[i][idxs]
    
    return pts

def identify_grasps(pts):
    outputs = infer_grasps(pl_model, pts, grid_size=pl_model.model.grid_size)

    class_logits, baseline_dir, approach_dir, grasp_offset, positions = outputs

    # only retain the information from the latest (-1) frame
    grasps = build_6dof_grasps(positions, baseline_dir, approach_dir, grasp_offset)
    confs = torch.sigmoid(class_logits)

    return grasps, confs, grasp_offset

def filter_grasps(grasps, confs, top_k):

    try: 
        len(confs.squeeze())
    except:
        breakpoint()

    # confidence thresholding
    grasps = grasps[confs.squeeze() > CONF_THRESHOLD]
    confs = confs.squeeze()[confs.squeeze() > CONF_THRESHOLD]

    if grasps.shape[0] == 0 or confs.shape[0] == 0:
        return None, None

    # top-k selection
    vals, top_idcs = torch.topk(confs.squeeze(), k=min(top_k, len(confs.squeeze())), sorted=True)
    grasps = grasps[top_idcs]
    confs = confs[top_idcs]

    if grasps.shape[0] == 0:
        return None, None

    return grasps, confs

@torch.inference_mode()
def find_grasps(pcl_msg: PointCloud2, top_k: int):
    global device

    with TimeIt("FIND_GRASPS() fn: "):
        with TimeIt('Unpack pointclouds'):
            try:
                with TimeIt("   ros_numpy: "):
                    pts = [ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl_msg, remove_nans=False).reshape(-1,3)]

                with TimeIt("   pts_to_gpu: "):
                    pts = [torch.from_numpy(pt.astype(np.float32)).to(device) for pt in pts]

            except ValueError as e:
                print(e)
                print("Is this error because there are fewer than 300x300 points?")
                return

            header = pcl_msg.header

        ## Processing pipeline

        # Start with pts, a list of Torch point clouds.

        # Downsample the points with uniform probability.
        with TimeIt('Downsample Points'):
            pts             = downsample_xyz(pts, PTS_PER_FRAME)
            if pts is None: return

        # Run the NN to identify grasp poses and confidences.
        with TimeIt('Find Grasps'):
            grasps, confs, widths = identify_grasps(pts)
            if pts is None or len(pts[-1]) == 0: return

        # Filter the grasps by thresholding and furthest-point sampling.
        with TimeIt('Filter Grasps'):
            grasps, confs   = filter_grasps(grasps, confs, top_k)

        if grasps is None: return

        return grasps, confs, widths

def homo_poses_to_ros_poses(homo_poses: torch.Tensor):
    """Convert homogeneous pose matrices to ros message poses

    Args:
        homo_poses (torch.Tensor): (4, 4, N) GPU tensor

    Returns:
        List[Pose]: ROS pose messages
    """

    # create numpy arrays for quaternions and vectors
    qs = rotation_matrix_to_quaternion(homo_poses[:,:3,:3].contiguous(), order = QuaternionCoeffOrder.XYZW).cpu().numpy()
    vs = homo_poses[:,:3,3].cpu().numpy()

    def q_v_to_pose(q, v):
        """Package up numpy arrays into a Pose message."""
        p = Pose()
        p.position.x, p.position.y, p.position.z = v
        (
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w 
        ) = q
        return p

    return [q_v_to_pose(q, v) for q, v in zip(qs, vs)]

def handle_find_grasps(req: FindGraspsRequest) -> FindGraspsResponse:
    """ Handle a FindGraspsRequest by identifying grasp poses in a point cloud,
    fulfilling the Service obligation from FindGrasps.srv.
    """
    # identify grasp poses, confidences, and widths
    with gpu_mtx:
        grasp_pose_mtxs, confs, widths = find_grasps(req.points, req.top_k.data)

    with TimeIt("Sending response: "):
        # create list of ROS poses from the homogenous pose matrices
        grasp_poses = homo_poses_to_ros_poses(grasp_pose_mtxs)

        # collect the grasp poses into a list of Grasp messages
        grasps_list = []
        for matrix, pose, width in zip(grasp_pose_mtxs, grasp_poses, widths):
            g = Grasp()
            g.pose = pose
            g.approach = Vector3(*matrix[:3, 2].cpu().numpy())
            g.baseline = Vector3(*matrix[:3, 0].cpu().numpy())
            g.width = width.cpu().numpy()
            grasps_list.append(g)

        # make a plural Grasps message containing the header and confidences
        grasps_msg = Grasps()
        grasps_msg.header = req.points.header
        grasps_msg.grasps = grasps_list
        grasps_msg.confs  = confs.cpu().numpy()

        return FindGraspsResponse(grasps_msg)

def find_grasps_server():
    rospy.init_node('find_grasps_server')
    s = rospy.Service('find_grasps', FindGrasps, handle_find_grasps)
    print("Ready to find grasps.")
    rospy.spin()
    
if __name__ == "__main__":
    find_grasps_server()