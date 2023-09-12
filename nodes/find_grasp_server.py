#! /home/playert/miniconda3/envs/tsgrasp/bin/python
# shebang is for the Python3 environment with all dependencies

# tsgrasp dependencies
import sys, os

print(f"Python is {sys.executable}")
sys.path.append("/home/playert/Research/grasp_synth_ws/src/grasp_synth/")
from nn.load_model import load_model

# ROS dependencies
import rospy
from geometry_msgs.msg import Pose, Vector3, PoseArray, PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from rospy.numpy_msg import numpy_msg
import ros_numpy
from tf.transformations import quaternion_matrix

from grasp_synth.msg import Grasp, Grasps, Grasps2
from grasp_synth.srv import FindGrasps, FindGraspsRequest, FindGraspsResponse
from utils import TFHelper

# python dependencies
import numpy as np
from collections import deque
import torch
from threading import Lock
import copy
from kornia.geometry.conversions import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion, QuaternionCoeffOrder
import math
import MinkowskiEngine as ME
from pytorch3d.ops import sample_farthest_points
from typing import List
# torch.backends.cudnn.benchmark=True # makes a big difference on FPS for some PTS_PER_FRAME values, but seems to increase memory usage and can result in OOM errors.

## global constants
QUEUE_LEN       = 1
PTS_PER_FRAME   = 45000
GRIPPER_DEPTH   = 0.12 # 0.1034 for panda
CONF_THRESHOLD  = 0
TOP_K           = 1000
WORLD_BOUNDS    = torch.Tensor([[-2, -2, -1], [2, 2, 1]]) # (xyz_lower, xyz_upper)
CAM_BOUNDS      = torch.Tensor([[-0.8, -0.8, 0.22], [0.8, 0.8, 2]]) # (xyz_lower, xyz_upper)

TF_ROLL, TF_PITCH, TF_YAW = 0, 0, math.pi/2
TF_X, TF_Y, TF_Z = 0, 0, 0


## load Pytorch Lightning network
device = torch.device('cuda')
pl_model = load_model().to(device)
pl_model.eval()

## Start node
rospy.init_node('grasp_server')

queue = deque(maxlen=QUEUE_LEN) # FIFO queue, right side is most recent
queue_mtx = Lock()
tf_helper = TFHelper()

unfiltered_grasp_pub = rospy.Publisher("unfiltered_grasps", Grasps2, queue_size=10)
filtered_grasps = None
filtered_grasp_pub = rospy.Publisher('filtered_grasps', Grasps, queue_size=10)
grasp_pose_pub = rospy.Publisher('filtered_grasp_poses', PoseArray, queue_size=10)

import time
class TimeIt:
    def __init__(self, s):
        self.s = s
        self.t0 = None
        self.t1 = None
        self.print_output = True

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, t, value, traceback):
        torch.cuda.synchronize()
        self.t1 = time.time()
        if self.print_output:
            print('%s: %s' % (self.s, self.t1 - self.t0))

# https://stackoverflow.com/questions/59387182/construct-a-rotation-matrix-in-pytorch
#@torch.jit.script
def eul_to_rotm(roll: float, pitch: float, yaw: float):
    """Convert euler angles to rotation matrix."""
    roll = torch.tensor([roll])
    pitch = torch.tensor([pitch])
    yaw = torch.tensor([yaw])

    tensor_0 = torch.zeros(1)
    tensor_1 = torch.ones(1)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                    torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    R = torch.mm(RZ, RY)
    R = torch.mm(R, RX)
    return R

#@torch.jit.script
def inverse_homo(tf):
    """Compute inverse of homogeneous transformation matrix.

    The matrix should have entries
    [[R,       Rt]
     [0, 0, 0, 1]].
    """
    R = tf[0:3, 0:3]
    t = R.T @ tf[0:3, 3].reshape(3, 1)
    return torch.cat([
        torch.cat([R.T, -t], dim=1),
        torch.tensor([[0, 0, 0, 1]]).to(R.device)
        ],dim=0
    )

def transform_to_eq_pose(poses):
    """Yaw by pi/2"""

    roll, pitch, yaw = TF_ROLL, TF_PITCH, TF_YAW
    x, y, z = TF_X, TF_Y, TF_Z
    tf = torch.cat([
        torch.cat([eul_to_rotm(roll, pitch, yaw), torch.Tensor([x, y, z]).reshape(3, 1)], dim=1),
        torch.Tensor([0, 0, 0, 1]).reshape(1, 4)
    ], dim=0).to(poses.device)
    return poses @ tf

def transform_vec(x: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
    """Transform 3D vector `x` by homogenous transformation `tf`.

    Args:
        x (torch.Tensor): (b, ..., 3) coordinates in R3
        tf (torch.Tensor): (b, 4, 4) homogeneous pose matrix

    Returns:
        torch.Tensor: (b, ..., 3) coordinates of transformed vectors.
    """

    x_dim = len(x.shape)
    assert all((
        len(tf.shape)==3,           # tf must be (b, 4, 4)
        tf.shape[1:]==(4, 4),       # tf must be (b, 4, 4)
        tf.shape[0]==x.shape[0],    # batch dimension must be same
        x_dim > 2                   # x must be a batched matrix/tensor
    )), "Argument shapes are unsupported."

    x_homog = torch.cat(
        [x, torch.ones(*x.shape[:-1], 1, device=x.device)], 
        dim=-1
    )
    
    # Pad the dimension of tf for broadcasting.
    # E.g., if x had shape (2, 3, 7, 3), and tf had shape (2, 4, 4), then
    # we reshape tf to (2, 1, 1, 4, 4)
    tf = tf.reshape(tf.shape[0], *([1]*(x_dim-3)), 4, 4)

    return (x_homog @ tf.transpose(-2, -1))[..., :3]

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
    print(coords.shape)

    with TimeIt("   tsgraspnet.model.forward"):
        class_logits, baseline_dir, approach_dir, grasp_offset = tsgraspnet.model.forward(stensor)

    ## Return the grasp predictions for the latest point cloud
    idcs = coords[:,1] == coords[:,1].max()
    return(
        class_logits[idcs], baseline_dir[idcs], approach_dir[idcs], grasp_offset[idcs], points[-1]
    )

#@torch.jit.script
def in_bounds(world_pts, BOUNDS):
    """Remove any points that are out of bounds"""
    x, y, z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]
    in_bounds = (
        (x > BOUNDS[0][0]) * 
        (y > BOUNDS[0][1]) * 
        (z > BOUNDS[0][2]) * 
        (x < BOUNDS[1][0]) *
        (y < BOUNDS[1][1]) * 
        (z < BOUNDS[1][2] )
    )
    return in_bounds

def bound_point_cloud_cam(pts, poses):
    """Bound the point cloud in the camera frame."""
    for i, pose in zip(range(len(pts)), poses):
        valid = in_bounds(pts[i], CAM_BOUNDS)
        pts[i] = pts[i][valid]
    
    ## ensure nonzero
    if sum(len(pt) for pt in pts) == 0:
        print("No points in bounds")
        return

    return pts

def bound_point_cloud_world(pts, poses):
    """Bound the point cloud in the world frame."""
    for i, pose in zip(range(len(pts)), poses):
        world_pc = transform_vec(
            pts[i].unsqueeze(0), pose.unsqueeze(0)
        )[0]
        valid = in_bounds(world_pc, WORLD_BOUNDS)
        pts[i] = pts[i][valid]
    
    ## ensure nonzero
    if sum(len(pt) for pt in pts) == 0:
        print("No points in bounds")
        return

    return pts

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

def transform_to_camera_frame(pts, poses):
    ## Transform all point clouds into the frame of the most recent image
    tf_from_cam_i_to_cam_N = inverse_homo(poses[-1]) @ poses
    pts =[
        transform_vec(
            pts[i].unsqueeze(0), 
            tf_from_cam_i_to_cam_N[i].unsqueeze(0)
        )[0]
        for i in range(len(pts))
    ]
    return pts

def identify_grasps(pts):

    try:
        outputs = infer_grasps(pl_model, pts, grid_size=pl_model.model.grid_size)

        class_logits, baseline_dir, approach_dir, grasp_offset, positions = outputs

        grasps = build_6dof_grasps(positions, baseline_dir, approach_dir, grasp_offset)

        confs = torch.sigmoid(class_logits)
    except Exception as e:
        print(e)
        breakpoint()
        print('debug')

    return grasps, confs, grasp_offset

def filter_grasps(grasps, confs, widths):

    # confidence thresholding
    grasps = grasps[confs.squeeze() > CONF_THRESHOLD]
    confs = confs.squeeze()[confs.squeeze() > CONF_THRESHOLD]

    if grasps.shape[0] == 0 or confs.shape[0] == 0:
        return None, None

    # top-k selection
    vals, top_idcs = torch.topk(confs.squeeze(), k=min(100*TOP_K, confs.squeeze().numel()), sorted=True)
    grasps = grasps[top_idcs]
    confs = confs[top_idcs]
    widths = widths[top_idcs]

    if grasps.shape[0] == 0:
        return None, None

    # furthest point sampling
    # # furthest point sampling by position
    pos = grasps[:,:3,3]
    _, selected_idcs = sample_farthest_points(pos.unsqueeze(0), K=TOP_K)
    selected_idcs = selected_idcs.squeeze()

    grasps = grasps[selected_idcs]
    confs = confs[selected_idcs]
    widths = widths[selected_idcs]

    return grasps, confs, widths

def ensure_grasp_y_axis_upward(grasps: torch.Tensor) -> torch.Tensor:
    """Flip grasps with their Y-axis pointing downwards by 180 degrees about the wrist (z) axis, 
        because we have mounted the camera on the wrist in the direction of the Y axis and don't 
        want it to be scraped off on the table.

    Args:
        grasps (torch.Tensor): (N, 4, 4) grasp pose tensor

    Returns:
        torch.Tensor: (N, 4, 4) grasp pose tensor with some grasps flipped
    """

    N = len(grasps)

    # The strategy here is to create a  Boolean tensor for whether
    # to flip the grasp. From the way we mounted our camera, we know that 
    # we'd prefer grasps with X axes that point up in the camera frame
    # (along the -Y axis). Therefore, we flip the rotation matrices of the
    # grasp poses that don't do that.

    # For speed, the flipping is done by allocating two (N, 4, 4) transformation
    # matrices: one for flipping (flips) and one for do-nothing (eyes). We select
    # between them with torch.where and perform matrix multiplication. This avoids 
    # a for loop (~100X speedup) at the expense of a bit of memory and obfuscation.

    y_axis = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    flip_about_z = torch.tensor([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=torch.float32, device=device)

    needs_flipping = grasps[:, :3, 1] @ y_axis > 0
    needs_flipping = needs_flipping.reshape(N, 1, 1).expand(N, 3, 3)

    eyes = torch.eye(3).repeat((N, 1, 1)).to(device)
    flips = flip_about_z.repeat((N, 1, 1)).to(device)

    tfs = torch.where(needs_flipping, flips, eyes)

    grasps[:,:3,:3] = torch.bmm(grasps[:,:3,:3], tfs)
    return grasps

def sort_grasps(grasps, confs, widths):
    vals, idcs = torch.sort(confs, descending=True)
    grasps = grasps[idcs]
    confs = confs[idcs]
    widths = widths[idcs]

    return grasps, confs, widths

    
@torch.inference_mode()
def find_grasps():
    global queue, device, queue_mtx

    print("in find_grasps")
    print(len(queue))

    # Wait for a frame to be available in the queue.
    r = rospy.Rate(5)
    with queue_mtx:
        n_frames = len(queue)
    while n_frames != QUEUE_LEN:
            print(f"Queue has length {len(queue)}")
            with queue_mtx:
                n_frames = len(queue)
            r.sleep()
    
    with TimeIt("FIND_GRASPS() fn: "):
        with TimeIt('Unpack pointclouds'):
            # only copy the queue with the mutex. Afterward, process the copy.
            with queue_mtx:
                if len(queue) != QUEUE_LEN: return
                q = copy.deepcopy(queue)
                # queue.clear()
                queue.popleft()

            try:
                msgs, poses = list(zip(*q))
                with TimeIt("   ros_numpy: "):
                    pts = [
                            ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False).reshape(-1,3)
                        for msg in msgs]

                # device = "cpu"
                with TimeIt("   pts_to_gpu: "):
                    pts = [torch.from_numpy(pt.astype(np.float32)).to(device) for pt in pts]
                poses = torch.Tensor(np.stack(poses)).to(device, non_blocking=True)
            except ValueError as e:
                print(e)
                print("Is this error because there are fewer than 300x300 points?")
                return

            header = msgs[-1].header

        ## Processing pipeline

        # Start with pts, a list of Torch point clouds.

        # Remove points that are outside of the boundaries in the camera frame.
        with TimeIt('Bound Point Cloud'):
            pts             = bound_point_cloud_cam(pts, poses)
            if pts is None or any(len(pcl) == 0 for pcl in pts): return

        # Remove points that are outside of the boundaries in the global frame.
        with TimeIt('Bound Point Cloud'):
            pts             = bound_point_cloud_world(pts, poses)
            if pts is None or any(len(pcl) == 0 for pcl in pts): return

        # Downsample the points with uniform probability.
        with TimeIt('Downsample Points'):
            pts             = downsample_xyz(pts, PTS_PER_FRAME)
            if pts is None or any(len(pcl) == 0 for pcl in pts): return

        # Transform the points into the frame of the last camera perspective.
        with TimeIt('Transform to Camera Frame'):
            pts             = transform_to_camera_frame(pts, poses)
            if pts is None or any(len(pcl) < 2 for pcl in pts): return # bug with length-one pcs

        # Run the NN to identify grasp poses and confidences.
        with TimeIt('Find Grasps'):
            grasps, confs, widths   = identify_grasps(pts)
            all_confs       = confs.clone() # keep the pointwise confs for plotting later

        # Filter the grasps by thresholding and (optionally) furthest-point sampling.
        with TimeIt('Filter Grasps'):
            grasps, confs, widths   = filter_grasps(grasps, confs, widths)

        if grasps is None: return

        with TimeIt('Ensure X Axis Upward'):
            grasps = ensure_grasp_y_axis_upward(grasps)

        with TimeIt('Transform to eq pose'):
            grasps = transform_to_eq_pose(grasps)

        grasps, confs, widths = sort_grasps(grasps, confs, widths)

        return grasps, confs, widths, header
        
# https://robotics.stackexchange.com/questions/20069/are-rospy-subscriber-callbacks-executed-sequentially-for-a-single-topic
# https://nu-msr.github.io/me495_site/lecture08_threads.html
def depth_callback(depth_msg):
    global queue, queue_mtx

    cam_tf_msg = tf_helper.get_transform("world", depth_msg.header.frame_id)

    cam_tf = torch.eye(4)
    cam_orn = cam_tf_msg.transform.rotation
    cam_orn = torch.Tensor([cam_orn.x, cam_orn.y, cam_orn.z, cam_orn.w])
    cam_orn = quaternion_to_rotation_matrix(cam_orn, order=QuaternionCoeffOrder.XYZW)
    cam_tf[:3, :3] = cam_orn

    cam_pos = cam_tf_msg.transform.translation
    cam_pos = torch.Tensor([cam_pos.x, cam_pos.y, cam_pos.z])
    cam_tf[:3, 3] = cam_pos 
    
    with queue_mtx:
        queue.append((depth_msg, cam_tf))

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
    global filtered_grasps, unfiltered_grasp_pub

    print("finding grasps!")
    # identify grasp poses, confidences, and widths
    grasp_pose_mtxs, confs, widths, header = find_grasps() # first apply top-400 filtering to send to kin_feas_filter

    with TimeIt("Sending response: "):
        # create list of ROS poses from the homogenous pose matrices
        grasp_poses = homo_poses_to_ros_poses(grasp_pose_mtxs)

        print(f"A: Conf min: {confs.min()}.\nConf max: {confs.max()}")

        # publish the grasps for kinematic feasibility filtering
        unfiltered_grasps = Grasps2()
        unfiltered_grasps.confs = confs.cpu().numpy()
        unfiltered_grasps.poses = grasp_poses
        unfiltered_grasps.widths = widths.cpu().numpy()
        unfiltered_grasps.header = header
        unfiltered_grasp_pub.publish(unfiltered_grasps)

        rospy.loginfo(f"Checking the feasibility of {len(unfiltered_grasps.poses)} grasps.")

        # wait for the filtered, kinematically feasible grasps to return
        r = rospy.Rate(5)
        timeout = rospy.Duration(15)
        start = rospy.Time.now()
        while filtered_grasps is None:
            if rospy.Time.now() - start > timeout:
                e = rospy.ServiceException(f"Did not filter grasps within {timeout.secs} seconds.")
                print(e)
                raise e
            r.sleep()

        rospy.loginfo(f"There are {len(filtered_grasps.poses)} kinematically feasible grasps. Keeping {min(len(filtered_grasps.poses), req.top_k)} of them.")

        print(f"B: Conf min: {np.min(filtered_grasps.confs)}.\nConf max: {np.max(filtered_grasps.confs)}")

        # collect the grasp poses into a list of Grasp messages
        grasps_list = []
        for pose, width in zip(filtered_grasps.poses[:req.top_k], filtered_grasps.widths[:req.top_k]):
            g = Grasp()
            g.pose = pose
            q = pose.orientation
            matrix = quaternion_matrix([q.x, q.y, q.z, q.y])
            g.approach = Vector3(*matrix[:3, 2])
            g.baseline = Vector3(*matrix[:3, 0])
            g.width = width
            grasps_list.append(g)

        # make a plural Grasps message containing the header and confidences
        grasps_msg = Grasps()
        grasps_msg.header = filtered_grasps.header
        grasps_msg.grasps = grasps_list
        grasps_msg.confs  = filtered_grasps.confs

        p = PoseArray()
        p.poses = filtered_grasps.poses
        p.header = filtered_grasps.header
        grasp_pose_pub.publish(p)

        filtered_grasps = None # reset filtered_grasps to be None for next time

        print(f"Grasps Response message:\n{grasps_msg}")

        filtered_grasp_pub.publish(grasps_msg)

        return FindGraspsResponse(grasps_msg)

def filtered_grasps_cb(msg):
    global filtered_grasps
    filtered_grasps = msg

def find_grasps_server():
    s = rospy.Service('find_grasps', FindGrasps, handle_find_grasps)
    pcl_sub = rospy.Subscriber('/trisect/stereo/points2', PointCloud2, depth_callback, queue_size=1) # /camera/depth/points
    filtered_grasp_sub = rospy.Subscriber('/filtered_grasps', Grasps2, filtered_grasps_cb, queue_size=1)

    print("Ready to find grasps.")
    rospy.spin()
    
if __name__ == "__main__":
    find_grasps_server()