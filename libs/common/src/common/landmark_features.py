"""Actual YuNet points as extra inputs, independent of the alignment template."""
import numpy as np
GEOMETRY_VERSION = 'yunet5_eye_frame_v1'
POINT_ORDER = ['right_eye','left_eye','nose','right_mouth','left_mouth']

def validate_geometry(values):
    g=np.asarray(values,dtype=np.float32)
    if g.shape!=(10,) or not np.isfinite(g).all():
        raise ValueError('Expected ten finite normalized landmark coordinates')
    if not np.allclose(g[:4],[-.5,0,.5,0],rtol=0,atol=1e-4):
        raise ValueError('Landmarks must use the declared eye-centered coordinate frame')
    if np.linalg.matrix_rank(g.reshape(5,2)-g.reshape(5,2).mean(0))<2:
        raise ValueError('Degenerate normalized landmark geometry')
    if np.abs(g).max()>10:
        raise ValueError('Implausible normalized landmark geometry')
    return np.ascontiguousarray(g)

def landmark_features(points):
    if points is None:raise ValueError('Five actual landmarks required; no zero/template fallback')
    p=np.asarray(points,dtype=np.float64)
    if p.shape!=(5,2) or not np.isfinite(p).all():raise ValueError('Five finite YuNet points required')
    delta=p[1]-p[0];distance=np.linalg.norm(delta)
    if distance<4:raise ValueError('Eye separation too small')
    if np.linalg.matrix_rank(p-p.mean(0))<2:raise ValueError('Degenerate landmarks')
    horizontal=delta/distance;vertical=np.array([-horizontal[1],horizontal[0]])
    centered=p-(p[0]+p[1])/2
    g=np.stack([centered@horizontal,centered@vertical],axis=1)/distance
    return validate_geometry(g.reshape(-1))

def flip_geometry(values):
    g=validate_geometry(values).reshape(5,2).copy()
    g[:,0]*=-1
    return validate_geometry(g[[1,0,2,4,3]].reshape(-1))
