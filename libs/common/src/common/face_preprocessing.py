"""Versioned RGB face preparation shared by training and inference."""
import numpy as np

BBOX = "bbox without margin or alignment"
ALIGNED = "yunet5_similarity_v1"
ALIGN_SIZE = 224
# Our fixed versioned template, NOT a claim about an optimal emotion template.
# YuNet order: right eye, left eye, nose, right mouth, left mouth.
TEMPLATE = np.array([[.32,.38],[.68,.38],[.50,.56],[.37,.75],[.63,.75]],dtype=np.float64) * (ALIGN_SIZE-1)


def similarity_matrix(points):
    """Least-squares rotation + uniform scale + translation, without reflection."""
    points=np.asarray(points,dtype=np.float64)
    if points.shape!=(5,2) or not np.isfinite(points).all():
        raise ValueError("Five finite YuNet points in full-frame pixels required")
    if np.linalg.norm(points[1]-points[0]) < 4:
        raise ValueError("Eye separation too small for reliable alignment")
    source=points-points.mean(0);target=TEMPLATE-TEMPLATE.mean(0)
    if np.linalg.matrix_rank(source)<2:
        raise ValueError("Degenerate landmark geometry")
    # Fit x'=a*x-b*y+tx; y'=b*x+a*y+ty.
    rows=[];values=[]
    for (x,y),(u,v) in zip(points,TEMPLATE):
        rows.extend([[x,-y,1,0],[y,x,0,1]]);values.extend([u,v])
    a,b,tx,ty=np.linalg.lstsq(np.asarray(rows),np.asarray(values),rcond=None)[0]
    if not np.isfinite([a,b,tx,ty]).all() or a*a+b*b < 1e-12:
        raise ValueError("Invalid alignment transform")
    return np.array([[a,-b,tx],[b,a,ty]],dtype=np.float64)


def prepare_face(rgb, bbox, landmarks=None, mode=BBOX):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or not rgb.size:
        raise ValueError("Expected non-empty uint8 RGB full frame")
    x,y,w,h=(int(v) for v in bbox)
    height,width=rgb.shape[:2]
    x1,y1=max(0,x),max(0,y);x2,y2=min(width,x+w),min(height,y+h)
    if w<=0 or h<=0 or x2<=x1 or y2<=y1:raise ValueError("Invalid bbox")
    if mode==BBOX:return np.ascontiguousarray(rgb[y1:y2,x1:x2])
    if mode!=ALIGNED:raise ValueError(f"Unsupported face preprocessing: {mode}")
    if landmarks is None:raise ValueError("Aligned model requires five landmarks; no bbox fallback")
    import cv2
    matrix=similarity_matrix(landmarks)
    # RGB channels stay RGB; OpenCV only warps, with replicated border pixels.
    return cv2.warpAffine(rgb,matrix,(ALIGN_SIZE,ALIGN_SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
