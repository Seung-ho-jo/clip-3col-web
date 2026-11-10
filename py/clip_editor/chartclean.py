"""도표 가장자리에 붙어 도표 뒤로 깔린 사진을 지운다 (배민·오리온·벤슨·농심 지면).

도표 바탕색을 찾고, 사진이 시작하는 열부터는 도표 글자 줄이 이어지는 만큼만 남긴다.
"""

from __future__ import annotations

import numpy as np, cv2

def chart_bg(rgb):
    q=(rgb//8).reshape(-1,3); v,c=np.unique(q,axis=0,return_counts=True)
    return (v[np.argmax(c)]*8+4).astype(np.int16)

def _photo_blocks(a, B, bs):
    h,w,_=a.shape; g=a.mean(2); near=np.abs(a-B).max(2)<14
    H,W=h//bs,w//bs; P=np.zeros((H,W),np.uint8)
    for i in range(H):
        for j in range(W):
            sl=(slice(i*bs,(i+1)*bs),slice(j*bs,(j+1)*bs))
            if near[sl].mean()>0.35: continue
            hist=np.histogram(g[sl],bins=16,range=(0,256))[0]
            if np.sort(hist)[-2:].sum()/hist.sum()>0.85: continue
            P[i,j]=1
    return P

def strip_edge_photo(rgb, ch, bs=16):
    """도표 한쪽 가장자리에 붙어 도표 뒤로 깔린 사진을 지운다. 사진이 없으면 None.

    사진이 시작하는 열(xb)을 찾고, xb 너머로는 도표 글자 줄이 이어지는 만큼만 도표로 본다.
    그 구간에서는 글자·꽉 찬 막대만 남기고 나머지를 도표 바탕색으로 칠한다.
    """
    a=rgb.astype(np.int16); h,w,_=a.shape; B=chart_bg(rgb)
    P=_photo_blocks(a,B,bs); H,W=P.shape
    if not H or not W: return None
    frac=cv2.blur(P.astype(np.float32),(3,1)).mean(0) if W>2 else P.mean(0)
    right=frac[-max(1,W//6):].mean() > frac[:max(1,W//6)].mean()
    f=frac if right else frac[::-1]
    if f[-max(1,W//6):].mean()<0.45: return None          # 가장자리에 사진이 없다
    jb=W
    for j in range(W-1,-1,-1):
        if f[j]>=0.2: jb=j
        else: break
    if jb>0.85*W or jb<0.2*W: return None
    flipped = not right
    if flipped: a=a[:, ::-1]; rgb=rgb[:, ::-1]
    xb=jb*bs
    g=a.mean(2); sat=a.max(2)-a.min(2)
    ink=((g<120)|((sat>90)&(g<180))).astype(np.uint8)
    # 축선처럼 긴 가로선은 떼어 낸다 (막대·글자가 축선으로 한 덩어리가 되지 않게). 축선은 xb 왼쪽만 남는다.
    hline=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,3*ch),np.uint8))
    hline=cv2.dilate(hline,np.ones((3,1),np.uint8))&ink
    ink=ink&~hline
    # 꽉 찬 막대는 따로 떼어 낸다 (글자와 붙어 한 덩어리가 되지 않게)
    n0,lab0,st0,_=cv2.connectedComponentsWithStats(ink,8)
    solid=np.zeros(n0,bool)
    for k in range(1,n0):
        x,y,ww,hh,area=st0[k]
        solid[k]=ww>=0.5*ch and hh>=ch and area>=0.85*ww*hh
    barmask=solid[lab0]
    # 자모 조각을 낱말 단위로 묶어서 본다 (한글 한 글자가 여러 조각으로 나뉘므로)
    k_=max(3,ch//4)|1
    txt=(ink>0)&~barmask
    grp=cv2.dilate(txt.astype(np.uint8),np.ones((5,k_),np.uint8))
    grp[barmask]=1
    grp[cv2.dilate(barmask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~barmask]=0
    n,lab,st,_=cv2.connectedComponentsWithStats(grp,8)
    lab=np.where(ink>0,lab,0)
    isbar=np.zeros(n,bool)
    for k in range(1,n0):
        if solid[k]:
            ys,xs=np.nonzero(lab0==k); isbar[lab[ys[0],xs[0]]]=True
    glyph=[k for k in range(1,n) if not isbar[k] and 0.35*ch<=st[k][3]<=1.8*ch and st[k][2]<=10*ch and st[k][4]>=4]
    bars=[k for k in range(1,n) if isbar[k]]
    rows=[(st[k][1],st[k][1]+st[k][3]) for k in glyph if st[k][0]+st[k][2]<=xb]
    last={}   # 글자 줄마다 마지막으로 남긴 글자의 오른쪽 끝 (표의 다음 칸까지는 넓게 건너뛴다)
    for k in glyph:
        if st[k][0]+st[k][2]<=xb:
            r=(st[k][1],st[k][1]+st[k][3]); last[r]=max(last.get(r,0),st[k][0]+st[k][2])
    jumped=set()
    keep_ids=set(); chart_end=xb
    def dark(k, near=False):
        # 글자: 진하거나 채도 높고, 둘레가 밝은 바탕이어야 한다 (사진 속 어두운 덩어리 제외)
        x,y,ww,hh,_=st[k]; m=lab[y:y+hh,x:x+ww]==k
        if not (g[y:y+hh,x:x+ww][m].mean()<110): return False
        if m.mean()>0.7: return False                    # 꽉 찬 덩어리는 글자가 아니다
        gg=g[y:y+hh,x:x+ww][m]; ss=sat[y:y+hh,x:x+ww][m]
        if (gg<75).sum()<0.4*max(1,(gg<120).sum()): return False   # 가장자리가 흐릿한 얼룩(사진 조각)은 글자가 아니다
        p=max(2,ch//4); y0,y1,x0,x1=max(0,y-p),min(h,y+hh+p),max(0,x-p),min(w,x+ww+p)
        ring=np.ones((y1-y0,x1-x0),bool); ring[y-y0:y-y0+hh,x-x0:x-x0+ww]=False
        return near or (g[y0:y1,x0:x1][ring]>150).mean()>=0.5
    axis=[st[k][1]+st[k][3] for k in bars if st[k][0]+st[k][2]<=xb]
    for k in sorted(glyph+bars,key=lambda k: st[k][0]):
        x,y,ww,hh,_=st[k]
        if x+ww<=xb: continue
        if k in bars:
            # 막대: 바닥이 도표 축(xb 왼쪽 막대들의 바닥)에 닿아야 한다
            if x-chart_end<=2*ch and any(abs(y+hh-b)<=ch//2 for b in axis):
                keep_ids.add(k); chart_end=max(chart_end,x+ww)
            continue
        if x<xb:                                          # 도표 쪽에서 시작해 걸친 낱말은 도표 것
            keep_ids.add(k); chart_end=max(chart_end,x+ww); last[(y,y+hh)]=x+ww; continue
        cy=y+hh/2
        hit=[r for r in list(last) if r[0]-ch/3<=cy<=r[1]+ch/3 and hh>=0.6*(r[1]-r[0])
             and x-last[r]<=(2.6*ch if (r in jumped or last[r]>xb) else 9*ch)]
        if hit and dark(k, near=min(x-last[r] for r in hit)<=0.6*ch):
            keep_ids.add(k); chart_end=max(chart_end,x+ww)
            for r in hit: last[r]=x+ww; jumped.add(r)
            last[(y,y+hh)]=x+ww; jumped.add((y,y+hh))
    x1=min(w,chart_end+ch//2)
    out=rgb.copy()
    keep=np.isin(lab,list(keep_ids)) if keep_ids else np.zeros((h,w),bool)
    keep=cv2.dilate(keep.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    zone=np.zeros((h,w),bool); zone[:,xb:]=True
    out[zone&~keep]=B.astype(np.uint8)
    out=out[:, :x1]
    if flipped: out=out[:, ::-1]
    return np.ascontiguousarray(out)
