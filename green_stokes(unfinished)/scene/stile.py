"""Geometria, palette e helper condivisi Green–Stokes. Manim CE 0.19.2 / Cairo / 16:9.

Geometry uses mathematical coordinates first;
shared mesh edges are keyed by vertex IDs, never inferred from screen distance.
"""
from collections import defaultdict
import numpy as np
from manim import (
    Scene, ThreeDScene, Group, VGroup, VMobject, Polygon, Line, Arrow, Line3D, Cone,
    Dot, DashedLine, NumberPlane, ArrowVectorField, ParametricFunction,
    MathTex, Tex, DecimalNumber, SurroundingRectangle, ValueTracker,
    FadeIn, FadeOut, Create, Write, GrowArrow, ReplacementTransform,
    Transform, TransformMatchingTex, LaggedStart, AnimationGroup, Indicate,
    always_redraw, config, linear, smooth,
    BLACK, WHITE, GREY_B, BLUE, BLUE_E, GREEN, YELLOW, RED, RED_A,
    TEAL, TEAL_B, TEAL_D, TEAL_E, MAROON_C, PURPLE, ORANGE,
    ORIGIN, RIGHT, LEFT, UP, DOWN, PI, DEGREES,
)

config.background_color = BLACK
np.random.seed(19)


def eq(*parts, size=36, width=12.5):
    obj = MathTex(*parts, font_size=size)
    if obj.width > width:
        obj.scale_to_fit_width(width)
    return obj


def caption(text, size=27, color=GREY_B):
    return Tex(text, font_size=size, color=color)


def framed(obj):
    return SurroundingRectangle(obj, buff=.24, stroke_width=2,
                                color=[ORANGE, TEAL_B])


def curve_point(t):
    r = 2 + .2 * np.sin(5*t)
    return np.array([r*np.cos(t), r*np.sin(t), 0.])


def curve_tangent(t):
    r, dr = 2+.2*np.sin(5*t), np.cos(5*t)
    return np.array([dr*np.cos(t)-r*np.sin(t), dr*np.sin(t)+r*np.cos(t), 0.])


def quadratic_field(p):
    x,y,_ = p
    return np.array([-y*y,x*x,0.])


def backdrop():
    return NumberPlane(background_line_style={"stroke_opacity":.22,
                       "stroke_width":1}, axis_config={"stroke_opacity":.25})


def soft_field(func, x_range=(-6,6,1), y_range=(-3,3,1)):
    return ArrowVectorField(func, x_range=list(x_range), y_range=list(y_range),
        colors=[BLUE,GREEN,YELLOW,RED], min_color_scheme_value=0,
        max_color_scheme_value=5, vector_config={"stroke_width":1.5}).set_opacity(.30)


def arrow(a,b,color=WHITE,width=2,tip=.12):
    return Arrow(a,b,buff=0,color=color,stroke_width=width,tip_length=tip,
                 max_tip_length_to_length_ratio=.2)


def domain_cells(h):
    """Center-sampled polyomino converging to the original polar domain."""
    result=[]
    k=int(np.ceil(2.3/h))
    for i in range(-k,k):
        for j in range(-k,k):
            x,y=(i+.5)*h,(j+.5)*h
            if np.hypot(x,y) < 2+.2*np.sin(5*np.arctan2(y,x)):
                result.append([(i,j),(i+1,j),(i+1,j+1),(i,j+1)])
    return result


def edge_incidence(faces):
    edges=defaultdict(list)
    for fi,face in enumerate(faces):
        for a,b in zip(face,face[1:]+face[:1]):
            key=tuple(sorted((a,b)))
            edges[key].append((a,b,fi))
    return edges


def planar_mesh(h, offset=ORIGIN, gain=1.):
    faces=domain_cells(h)
    pos=lambda v: np.array([h*v[0],h*v[1],0.])*gain+offset
    tiles=VGroup(*[Polygon(*[pos(v) for v in face],stroke_width=.7,
        stroke_color=TEAL_D,stroke_opacity=.5,fill_color=TEAL_E,
        fill_opacity=.16 if i%2 else .23) for i,face in enumerate(faces)])
    edges=edge_incidence(faces)
    boundary=VGroup(*[Line(pos(v[0][0]),pos(v[0][1]),color=TEAL,
        stroke_width=3) for v in edges.values() if len(v)==1])
    return tiles,boundary,faces,edges,pos


def surface_z(x,y,height=1.):
    return height*(.65-.13*(x*x+y*y)+.15*x)


def surface_normal(x,y,height=1.):
    n=np.array([height*(.26*x-.15),height*.26*y,1.])
    return n/np.linalg.norm(n)


def surface_topology(rings=4,sectors=40):
    xy={0:np.zeros(2)}
    vertex=lambda r,j:1+(r-1)*sectors+j%sectors
    for r in range(1,rings+1):
        for j in range(sectors):
            xy[vertex(r,j)]=curve_point(2*PI*j/sectors)[:2]*(r/rings)
    faces=[]
    for j in range(sectors):
        faces.append([0,vertex(1,j),vertex(1,j+1)])
    for r in range(1,rings):
        for j in range(sectors):
            faces.append([vertex(r,j),vertex(r+1,j),vertex(r+1,j+1),vertex(r,j+1)])
    return xy,faces


def surface_mesh(height=1., rings=4, sectors=40):
    xy,faces=surface_topology(rings,sectors)
    pos=lambda v:np.r_[xy[v],surface_z(*xy[v],height)]
    tiles=VGroup(*[Polygon(*[pos(v) for v in face],stroke_color=TEAL_D,
        stroke_width=.65,stroke_opacity=.65,fill_color=TEAL_E,
        fill_opacity=.18 if i%2 else .26) for i,face in enumerate(faces)])
    return tiles,xy,faces,pos


def lifted_curve(height=1.,color=TEAL,width=3):
    def f(t):
        p=curve_point(t)
        p[2]=surface_z(p[0],p[1],height)+.012
        return p
    return ParametricFunction(f,t_range=[0,2*PI,.025],color=color,stroke_width=width)


def surface_edge(a,b,height=1.,color=WHITE,inset=None,width=1.5):
    a,b=np.array(a),np.array(b)
    if inset is not None:
        a=.89*a+.11*inset; b=.89*b+.11*inset
    pa=np.r_[a,surface_z(*a,height)+.025]
    pb=np.r_[b,surface_z(*b,height)+.025]
    d=pb-pa; length=np.linalg.norm(d); u=d/length
    normal=surface_normal(*((a+b)/2),height)
    side=np.cross(normal,u); side/=np.linalg.norm(side)
    base=pa+.65*d; tip=pa+.83*d
    half=min(.06,.11*length)
    return VGroup(Line(pa,pb,color=color,stroke_width=width),
        Polygon(tip,base+half*side,base-half*side,color=color,
                fill_opacity=1,stroke_width=0))


def lowpoly_arrow(start,end,color=TEAL,resolution=6,thickness=.015,height=.12,base_radius=.045):
    """Same solid-arrow appearance without Arrow3D's default dense cone."""
    start,end=np.array(start),np.array(end)
    direction=(end-start)/np.linalg.norm(end-start)
    shaft=Line3D(start,end-height*direction,thickness=thickness,color=color,resolution=resolution)
    tip=Cone(direction=direction,base_radius=base_radius,height=height,
             resolution=(2,8),fill_color=color,fill_opacity=1,stroke_width=0).shift(end)
    return VGroup(shaft,tip)
