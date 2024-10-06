import open3d as o3d
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from numpy.linalg import eigh

def create_custom_pointcloud(plydata,mode): # 파일 경로 지정 시
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    f_dc = np.squeeze(features_dc, axis=2)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # open3d의 PointCloud 객체 생성 및 데이터 설정
    pcd1 = o3d.t.geometry.PointCloud()
    pcd2 = o3d.t.geometry.PointCloud()
    pcd3 = o3d.t.geometry.PointCloud()
    pcd4 = o3d.t.geometry.PointCloud()

    # pcl1 : xyz, color, opacity 모두 존재
    # pcl2 : color,opacity 만 존재 -> color,opacity만을 이용해 clustering 진행
    # pcl3 : xyz, opacity만 존재 -> xyz,opacity만을 이용해 clustering 진행
    # pcl4 : xyz, color만 존재 -> xyz, color만을 이용해 clustering 진행

    # pcd1 : 위치, 색상, 투명도
    # pcd2 : 색상, 투명도
    # pcd3 : 위치, 투명도
    # pcl4 : 위치, 색상

    pcd1.point.positions = xyz
    pcd1.point.colors = f_dc
    pcd1.point.opa = opacities

    xyz_zero = np.zeros_like(xyz)
    pcd2.point.positions = xyz_zero
    pcd2.point.colors = f_dc
    pcd2.point.opa = opacities

    f_dc_zero = np.zeros_like(f_dc)
    pcd3.point.positions = xyz
    pcd3.point.colors = f_dc_zero
    pcd3.point.opa = opacities

    opa_zero = np.zeros_like(opacities)
    pcd4.point.positions = xyz
    pcd4.point.colors = f_dc
    pcd3.point.opa = opa_zero

    return pcd1, pcd2, pcd3, pcd4


def set_parameter(gaussian,mode):
    xyz = gaussian._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussian._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussian._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussian._opacity.detach().cpu().numpy()
    scale = gaussian._scaling.detach().cpu().numpy()
    rotation = gaussian._rotation.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in gaussian.construct_list_of_attributes()]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    
    # xyz = gaussian._xyz.detach().cpu().numpy()
    # features_dc = gaussian._features_dc.detach().cpu().numpy()
    # f_dc = np.squeeze(features_dc, axis=2)

    plydata = PlyData([el])
    pcd1, pcd2, pcd3, pcd4 = create_custom_pointcloud(plydata,mode)
    return pcd1, pcd2, pcd3, pcd4


# DBSCAN_Clustering
def dbscan(pcl,eps , min_points):
    pcl = pcl.cuda(0)
    # labels = np.array(pcl.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True).cpu())
    labels = pcl.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True).cpu().numpy()
    if labels.size == 0:
        print("No clusters found. DBSCAN did not return any labels.")
        return labels
    
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    return labels


def cluster_mean(pcl1,pcl2,labels):
    data = []
    max_label = labels.max()
    num_clusters = max_label + 1

    # points = np.asarray(pcl1.points.cpu())
    # colors = np.asarray(pcl2.colors.cpu())
    points = pcl1.point.positions.numpy()
    colors = pcl2.point.colors.numpy()

    # 클러스터의 대표값 계산
    unique_labels = set(labels) - {-1}  # -1은 노이즈로 간주하여 제외
    print(f"Number of clusters: {len(unique_labels)}")

    cluster_centers = []

    for label in unique_labels:
        cluster_points = points[labels == label]
        cluster_colors = colors[labels == label]

        # 클러스터 중심 좌표 계산 (평균 좌표)
        center_coords = np.mean(cluster_points, axis=0)

        # 클러스터 평균 색상 계산
        mean_color = np.mean(cluster_colors, axis=0)

        cluster_centers.append((center_coords, mean_color))

    # 노이즈 포인트 처리
    noise_points = points[labels == -1]
    if noise_points.size > 0:
        print(f"Number of noise points: {len(noise_points)}")

    xyz = np.asarray(cluster_centers)[:,0]
    xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    feature_dc = np.asarray(cluster_centers)[:,1][:,np.newaxis,:]
    feature_dc = nn.Parameter(torch.tensor(feature_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))

    return points, xyz, feature_dc

def opacities_cluster(pcd, labels, mode):
  opacities = pcd.point.opa.numpy()
  unique_labels = np.unique(labels)
  unique_labels = set(labels) - {-1}

  if mode == 4:
    opacity = []
    for label in unique_labels:
      opacity.append(opacities[labels == label])
    
    max_opacities = []
    for i in opacity:
      max_opacities.append([i.max()])   
    opacities = nn.Parameter(torch.tensor(max_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    return opacities
  else:
    cluster_centers = []
    for label in unique_labels:
      cluster_opacities = opacities[labels == label]
      mean_opacity = np.mean(cluster_opacities, axis=0)
      cluster_centers.append((mean_opacity))
    opacities = np.asarray(cluster_centers)[:,0][..., np.newaxis]
    opacities = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))  
    return opacities
    
def fit_gaussian_to_cluster(points):
    """
    주어진 포인트 클라우드에 대해 가우시안을 맞추고,
    스케일과 회전 값을 반환합니다.

    Args:
    - points (np.ndarray): (N, 3) 크기의 포인트 클라우드 배열.

    Returns:
    - scales (np.ndarray): (3,) 크기의 축 스케일 (고유값의 제곱근).
    - rotation_matrix (np.ndarray): (#, 4) 크기의 회전 행렬.
    """

    # 클러스터 중심 (평균) 계산
    mean = np.mean(points, axis=0)

    if points.shape[0] == 1:
        # 포인트가 하나일 경우: 스케일을 최소값으로, 회전을 단위 행렬로 설정
        scales = np.array([1e-6, 1e-6, 1e-6])
        rotation_quaternion = np.array([1, 0, 0, 0])
    else:
        # 클러스터 중심으로부터의 편차 계산
        demeaned_points = points - mean

        # 공분산 행렬 계산
        covariance_matrix = np.cov(demeaned_points, rowvar=False)

        # 공분산 행렬의 고유값 및 고유벡터 계산
        eigenvalues, eigenvectors = eigh(covariance_matrix)

        # 고유값의 제곱근은 스케일을 나타냄
        scales = np.sqrt(eigenvalues)
        # nan값 처리
        scales = np.where(np.isnan(scales), 1e-6, scales)

        # 고유벡터는 회전 행렬을 형성
        rotation_matrix = eigenvectors

        # 회전 행렬을 쿼터니언으로 변환
        rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()
  
    return scales, rotation_quaternion

def scale_rotaion(points,labels):
  unique_labels = set(labels) - {-1}
  scales_array = []
  rotation_array = []
  for label in unique_labels:
      cluster_points = points[labels == label]
      # cluster_colors = colors[labels == label]

      scales, rotation_matrix = fit_gaussian_to_cluster(cluster_points)
      scales_array.append(scales)
      rotation_array.append(rotation_matrix)
  scales = np.asarray(scales_array)
  scales = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))

  rots = np.asarray(rotation_array)
  rots = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

  return scales, rots
