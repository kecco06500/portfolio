"""Mathematical checks for edge cancellation, orientation and local error."""
import json
import numpy as np
from scene.stile import (domain_cells,edge_incidence,surface_topology,
    surface_normal,surface_z,quadratic_field)

def integral_edge(a,b,field):
    nodes,weights=np.polynomial.legendre.leggauss(4)
    delta=b-a
    return sum(w*np.dot(field(a+(t+1)*delta/2),delta)/2 for t,w in zip(nodes,weights))

def check():
    results={}
    areas=[]
    for h in [.6,.3,.15]:
        faces=domain_cells(h);edges=edge_incidence(faces)
        assert all(len(v) in [1,2] for v in edges.values())
        for inc in edges.values():
            if len(inc)==2:
                assert inc[0][0]==inc[1][1] and inc[0][1]==inc[1][0]
        p=lambda v:np.r_[np.array(v)*h,0.]
        boundary=sum(integral_edge(p(v[0][0]),p(v[0][1]),quadratic_field) for v in edges.values() if len(v)==1)
        surface=sum((2*np.mean([p(v)[0] for v in f])+2*np.mean([p(v)[1] for v in f]))*h*h for f in faces)
        assert abs(boundary-surface)<1e-9,(h,boundary,surface)
        areas.append(dict(h=h,cells=len(faces),area=len(faces)*h*h,error=abs(len(faces)*h*h-4.02*np.pi),green_residual=abs(boundary-surface)))
    results['green_mesh']=areas
    for h in [1.2,.6,.3,.15]:
        v=[np.array([1,.5,0]),np.array([1+h,.5,0]),np.array([1+h,.5+h,0]),np.array([1,.5+h,0])]
        circulation=sum(integral_edge(a,b,quadratic_field) for a,b in zip(v,v[1:]+v[:1]))
        assert abs(circulation/h**2-(3+2*h))<1e-10
    results['local_limit']='Exact circulation/h² = 3+2h verified at all animated sizes.'
    xy,faces=surface_topology();edges=edge_incidence(faces)
    for inc in edges.values():
        assert len(inc) in (1,2)
        if len(inc)==2:assert inc[0][0]==inc[1][1] and inc[0][1]==inc[1][0]
    area=0
    for f in faces:
        vs=[xy[v] for v in f]
        signed=sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(vs,vs[1:]+vs[:1]))/2
        assert signed>0
        area+=signed
    field=lambda p:np.array([-p[1],p[0],0.])
    p=lambda v:np.r_[xy[v],surface_z(*xy[v],1.4)]
    line=sum(integral_edge(p(inc[0][0]),p(inc[0][1]),field) for inc in edges.values() if len(inc)==1)
    assert abs(line-2*area)<1e-9
    for x,y in [(.9,-.45),(1,1),(-1,.5)]:
        h=1e-5
        gx=(surface_z(x+h,y,1.4)-surface_z(x-h,y,1.4))/(2*h)
        gy=(surface_z(x,y+h,1.4)-surface_z(x,y-h,1.4))/(2*h)
        n=np.array([-gx,-gy,1]);n/=np.linalg.norm(n)
        assert np.linalg.norm(n-surface_normal(x,y,1.4))<1e-9
    # The highlighted adjacent cells must actually share a reversed edge.
    shared=[v for v in edge_incidence([faces[75],faces[76]]).values() if len(v)==2]
    assert len(shared)==1
    results['stokes_mesh']=dict(faces=len(faces),boundary_edges=sum(len(v)==1 for v in edges.values()),stokes_residual=abs(line-2*area),normals='finite-difference gradient verified',selected_shared_edges=len(shared))
    return results

if __name__=='__main__':print(json.dumps(check(),indent=2))
