"""Raccordi e capitoli finali: Green globale, Stokes, 1D e forme differenziali."""
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

from .stile import (eq, caption, framed, curve_point, curve_tangent, quadratic_field, backdrop, soft_field, arrow, domain_cells, edge_incidence, planar_mesh, surface_z, surface_normal, surface_topology, surface_mesh, lifted_curve, surface_edge, lowpoly_arrow)

class CirculationIntegral(Scene):
    """Insert after SumDotProducts: a weighted sum, then the line integral."""
    def construct(self):
        plane=backdrop(); field=soft_field(quadratic_field)
        curve=ParametricFunction(curve_point,t_range=[0,2*PI,.02],color=TEAL,stroke_width=3)
        dot=Dot(curve_point(0),color=YELLOW,radius=.045)
        tv=arrow(curve_point(0),curve_point(0)+.4*curve_tangent(0),TEAL)
        fv=arrow(curve_point(0),curve_point(0)+.4*quadratic_field(curve_point(0)),MAROON_C)
        graph=VGroup(plane,field,curve)
        self.add(graph,dot,tv,fv); self.wait(1)
        self.play(FadeOut(VGroup(dot,tv,fv)),graph.animate.scale(.72).shift(LEFT*3),run_time=1.5)
        point=lambda t:.72*curve_point(t)+LEFT*3
        def samples(n):
            ts=np.linspace(0,2*PI,n+1)
            return VGroup(*[arrow(point(a),point(b),TEAL,width=2,tip=.07)
                for a,b in zip(ts[:-1],ts[1:])])
        chords=samples(12)
        title=caption(r"Dal contributo locale alla circuitazione").move_to([0,3.25,0])
        local=eq(r'\vec F(\gamma(t_i))',r'\cdot',r"\gamma'(t_i)",r'\Delta t',size=35,width=5.7).move_to([3.1,1.5,0])
        local[0].set_color(MAROON_C); local[2].set_color(TEAL);local[3].set_color(YELLOW)
        sample=Dot(point(.6),color=YELLOW,radius=.045)
        self.play(Write(title),LaggedStart(*[GrowArrow(a) for a in chords],lag_ratio=.05),FadeIn(sample),run_time=2)
        self.play(Write(local));self.wait(2)
        sum_eq=eq(r'\sum_{i=0}^{N-1}\vec F(\gamma(t_i))\cdot\gamma\prime(t_i)\,\Delta t',size=32,width=5.8).move_to([3.1,0,0])
        self.play(Write(sum_eq));self.wait(1.5)
        small=eq(r'\Delta t\longrightarrow0',size=30).move_to([3.1,-1.05,0])
        medium=samples(24)
        self.play(Write(small),ReplacementTransform(chords,medium),run_time=1.5)
        dense=samples(48)
        self.play(ReplacementTransform(medium,dense),run_time=1.3)
        integral=eq(r'\oint_{\gamma}\vec F\cdot d\vec r',r'=',
            r'\int_0^{2\pi}\vec F(\gamma(t))\cdot\gamma\prime(t)\,dt',size=31,width=6.1).move_to([3.05,-2.1,0])
        self.play(Write(integral),run_time=2);self.play(Create(framed(integral)));self.wait(3)
        self.play(FadeOut(VGroup(title,local,sum_eq,small,sample,dense)),run_time=1)
        self.wait(1)


class LocalLimit(Scene):
    """Same nonlinear field as the existing curve. Exact, nonzero error."""
    def construct(self):
        origin=np.array([-4.1,-.7,0.]); gain=1.35
        xy=lambda x,y:origin+gain*np.array([x-1,y-.5,0.])
        h=ValueTracker(1.2)
        field=soft_field(quadratic_field,x_range=(-1,4,.6),y_range=(-2,3,.6))
        field.scale(gain).shift(origin-gain*np.array([1,.5,0.]))
        field.set_opacity(.24)
        square=always_redraw(lambda:Polygon(xy(1,.5),xy(1+h.get_value(),.5),
            xy(1+h.get_value(),.5+h.get_value()),xy(1,.5+h.get_value()),
            color=TEAL,stroke_width=3,fill_color=TEAL_E,fill_opacity=.15))
        p=Dot(origin,color=YELLOW,radius=.045)
        p_label=eq(r'p=(1,\tfrac12)',size=26).next_to(p,DOWN,buff=.25)
        field_label=eq(r'\vec F=(-y^2,x^2)',size=32).move_to([-3,2.5,0])
        title=caption(r"Il quadrato si restringe; la densit\`a converge").move_to([0,3.3,0])
        hlabel=eq('h=',size=30).move_to([-3,-1.7,0])
        number=DecimalNumber(h.get_value(),num_decimal_places=2,font_size=30,color=TEAL).next_to(hlabel,RIGHT)
        number.add_updater(lambda m:m.set_value(h.get_value()).next_to(hlabel,RIGHT))
        curl=eq(r'(\nabla\times\vec F)_z(p)=2x_p+2y_p=3',size=33,width=6).move_to([2.65,1.9,0])
        density=eq(r'\frac{1}{h^2}\oint_{\partial R_h}\vec F\cdot d\vec r',r'=3+2h',size=35,width=6).move_to([2.65,.45,0])
        density[1].set_color(TEAL)
        result_label=eq('=',size=36).move_to([1.9,-.7,0])
        result=DecimalNumber(5.4,num_decimal_places=2,font_size=38,color=TEAL).next_to(result_label,RIGHT)
        result.add_updater(lambda m:m.set_value(3+2*h.get_value()).next_to(result_label,RIGHT))
        self.play(FadeIn(field),Create(square),FadeIn(p),Write(p_label),Write(field_label),Write(title),run_time=2)
        self.play(Write(curl),Write(hlabel),FadeIn(number));self.wait(1)
        self.play(Write(density),Write(result_label),FadeIn(result),run_time=2);self.wait(1)
        for value in [.6,.3,.15]:
            self.play(h.animate.set_value(value),run_time=1.5);self.wait(1)
        limit=eq(r'\lim_{h\to0}\frac{1}{h^2}\oint_{\partial R_h}\vec F\cdot d\vec r',r'=',r'(\nabla\times\vec F)_z(p)',size=34,width=11.8).move_to([0,-2.75,0])
        limit[2].set_color(PURPLE)
        self.play(Write(limit),run_time=2);self.play(Create(framed(limit)));self.wait(3)
        number.clear_updaters();result.clear_updaters();square.clear_updaters()


class GreenGlobal(Scene):
    """Exact discrete edge cancellation, then a boundary-convergent mesh."""
    def construct(self):
        offset=np.array([-3.,.25,0.]); gain=1.12
        faces=[[(i,j),(i+1,j),(i+1,j+1),(i,j+1)] for i in range(3) for j in range(2)]
        pos=lambda v: offset+np.array([(v[0]-1.5)*1.2,(v[1]-1)*1.2,0.])
        cells=VGroup(*[Polygon(*[pos(v) for v in face],stroke_width=1,
            stroke_color=TEAL_D,fill_color=TEAL_E,fill_opacity=.13) for face in faces])
        edge_map=edge_incidence(faces); currents=VGroup(); inner=VGroup();outer=VGroup()
        for incidents in edge_map.values():
            for a,b,fi in incidents:
                c=np.mean([pos(v) for v in faces[fi]],axis=0)
                m=arrow(.84*pos(a)+.16*c,.84*pos(b)+.16*c,
                        MAROON_C if len(incidents)==2 else TEAL,1.8,.11)
                currents.add(m); (inner if len(incidents)==2 else outer).add(m)
        title=caption(r"I lati interni si cancellano").move_to([0,3.25,0])
        sum_local=eq(r'\sum_k\oint_{\partial R_k}\vec F\cdot d\vec r',size=38,width=5.5).move_to([3.1,1.25,0])
        pair=eq(r'\int_e\vec F\cdot d\vec r',r'+',r'\int_{-e}\vec F\cdot d\vec r',r'=0',size=32,width=5.9).move_to([3.1,-.1,0])
        pair[0].set_color(MAROON_C);pair[2].set_color(MAROON_C)
        self.play(Write(title),LaggedStart(*[FadeIn(c) for c in cells],lag_ratio=.10),run_time=2)
        self.play(LaggedStart(*[GrowArrow(a) for a in currents],lag_ratio=.015),Write(sum_local),run_time=2)
        self.play(Write(pair),run_time=1.5);self.wait(1.5)
        self.play(LaggedStart(*[FadeOut(a) for a in inner],lag_ratio=.025),run_time=2)
        outer_eq=eq(r'=\oint_{\partial D_h}\vec F\cdot d\vec r',size=38,width=5.5).move_to([3.1,-1.25,0]);outer_eq.set_color(TEAL)
        self.play(Write(outer_eq),run_time=1.5);self.wait(2)
        curve=ParametricFunction(lambda t:gain*curve_point(t)+offset,t_range=[0,2*PI,.02],color=TEAL,stroke_width=3)
        tiles,boundary,*_=planar_mesh(.6,offset,gain)
        newtitle=caption(r"Dai quadratini a una regione curva").move_to(title)
        self.play(FadeOut(VGroup(cells,outer,pair)),ReplacementTransform(title,newtitle),Create(curve),FadeIn(tiles),Create(boundary),run_time=2)
        gamma=eq(r'\partial D',size=29).next_to(curve,UP,buff=.15)
        riemann=eq(r'\approx\sum_k',r'(Q_x-P_y)(p_k)',r'\,\Delta A_k',size=35,width=5.7).move_to([3.1,.05,0])
        riemann[1].set_color(PURPLE);riemann[2].set_color(TEAL)
        riemann.move_to([3.1,-1.25,0])
        self.play(outer_eq.animate.move_to([3.1,.05,0]),Write(gamma),Write(riemann),run_time=1.5);self.wait(1.5)
        hlabel=eq(r'h=0.60',size=27).move_to([-3,-2.75,0])
        self.play(Write(hlabel))
        for h in [.3,.15]:
            nt,nb,*_=planar_mesh(h,offset,gain)
            nh=eq(f'h={h:.2f}',size=27).move_to(hlabel)
            self.play(ReplacementTransform(tiles,nt),ReplacementTransform(boundary,nb),ReplacementTransform(hlabel,nh),run_time=2)
            tiles,boundary,hlabel=nt,nb,nh;self.wait(1.2)
        note=eq(r'h\to0:\quad D_h\to D',size=29).move_to(hlabel)
        exact_fill=Polygon(*[gain*curve_point(t)+offset for t in np.linspace(0,2*PI,160,endpoint=False)],
            stroke_width=0,fill_color=TEAL_E,fill_opacity=.18).set_z_index(-1)
        self.play(ReplacementTransform(hlabel,note),FadeOut(boundary),FadeOut(tiles),FadeIn(exact_fill),run_time=1.3)
        global_eq=eq(r'\oint_{\partial D}(P\,dx+Q\,dy)',r'=',r'\iint_D\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dA',size=39,width=12).move_to([0,-2.75,0])
        global_eq[0].set_color(TEAL);global_eq[2].set_color(PURPLE)
        self.play(FadeOut(VGroup(sum_local,outer_eq,riemann,note)),Write(global_eq),run_time=2.5)
        name=caption(r"Teorema di Green",size=32,color=WHITE).move_to(newtitle)
        self.play(ReplacementTransform(newtitle,name),Create(framed(global_eq)),run_time=1.2)
        self.wait(4)
        self.play(FadeOut(VGroup(exact_fill,gamma,name,global_eq,*[m for m in self.mobjects if isinstance(m,SurroundingRectangle)])),Transform(curve,ParametricFunction(curve_point,t_range=[0,2*PI,.025],color=TEAL,stroke_width=3)),run_time=2)
        self.wait(1)


class StokesSurface(ThreeDScene):
    """Plane→curved surface, tangent-normal projection and boundary sum."""
    def construct(self):
        self.set_camera_orientation(phi=0,theta=-90*DEGREES,zoom=1)
        curve=lifted_curve(0)
        mesh,xy,faces,pos=surface_mesh(0)
        self.add(curve);self.wait(1)
        self.play(FadeIn(mesh),run_time=1.5)
        curved_mesh,_,_,_=surface_mesh(1.4)
        curved_boundary=lifted_curve(1.4)
        self.move_camera(phi=60*DEGREES,theta=-45*DEGREES,zoom=1.4,run_time=3,
                         added_anims=[Transform(mesh,curved_mesh),Transform(curve,curved_boundary)])
        title=caption(r"La stessa cancellazione, su una superficie curva",size=29).to_edge(UP,buff=.35)
        self.add_fixed_in_frame_mobjects(title);self.play(FadeIn(title));self.wait(2)
        normals=VGroup()
        for x,y in [(-1.1,-.8),(.2,-1.2),(1.25,-.35),(.8,.8),(-.5,1.1),(-.3,-.1)]:
            p=np.array([x,y,surface_z(x,y,1.4)])
            normals.add(lowpoly_arrow(p,p+.65*surface_normal(x,y,1.4),color=TEAL,resolution=6,thickness=.015,height=.12,base_radius=.045))
        normal_eq=eq(r'\hat n=\hat n(p)',size=32).to_corner(UP+LEFT,buff=.55).shift(DOWN*.65)
        self.add_fixed_in_frame_mobjects(normal_eq)
        self.play(LaggedStart(*[FadeIn(n) for n in normals],lag_ratio=.1),FadeIn(normal_eq),run_time=2);self.wait(2)
        pxy=np.array([1.4,.5]);p=np.r_[pxy,surface_z(*pxy,1.4)+.035]
        n=surface_normal(*pxy,1.4)
        curl=np.array([0,0,1.55]); projection=np.dot(curl,n)*n
        curl_arrow=lowpoly_arrow(p,p+curl,color=PURPLE,resolution=8,thickness=.022,height=.15,base_radius=.055)
        projected=lowpoly_arrow(p,p+projection,color=TEAL,resolution=8,thickness=.022,height=.15,base_radius=.055)
        guide=DashedLine(p+curl,p+projection,color=WHITE,stroke_width=1.4,dash_length=.07)
        tile=Polygon(*[np.array([x,y,surface_z(x,y,1.4)+.025]) for x,y in [(1.15,.25),(1.65,.25),(1.65,.75),(1.15,.75)]],color=YELLOW,fill_color=YELLOW,fill_opacity=.15,stroke_width=2)
        local=eq(r'\oint_{\partial S_k}\vec F\cdot d\vec r',r'\approx',r'(\nabla\times\vec F)',r'\cdot\hat n_k',r'\,\Delta S_k',size=36,width=11.7).to_edge(DOWN,buff=.5)
        local[2].set_color(PURPLE);local[3].set_color(TEAL);local[4].set_color(YELLOW)
        vector_label=eq(r'\nabla\times\vec F',r'\quad\longmapsto\quad',r'[(\nabla\times\vec F)\cdot\hat n]\hat n',size=29,width=8.7).move_to([-1.4,2.5,0])
        vector_label[0].set_color(PURPLE);vector_label[2].set_color(TEAL)
        self.add_fixed_in_frame_mobjects(local,vector_label)
        self.play(FadeOut(normals),FadeOut(normal_eq),FadeIn(tile),FadeIn(curl_arrow),FadeIn(vector_label),run_time=1.5)
        self.play(Create(guide),FadeIn(projected),Write(local),run_time=2.5);self.wait(3)
        self.play(FadeOut(VGroup(tile,curl_arrow,projected,guide,vector_label,local)),run_time=1.5)
        # Two adjacent surface cells share a radial edge with opposite incidences.
        selected=[faces[75],faces[76]]
        selected_map=edge_incidence(selected)
        patch_loops=VGroup();shared=VGroup()
        for inc in selected_map.values():
            for a,b,fi in inc:
                center=np.mean([xy[v] for v in selected[fi]],axis=0)
                m=surface_edge(xy[a],xy[b],1.4,MAROON_C if len(inc)==2 else WHITE,center,width=2)
                patch_loops.add(m)
                if len(inc)==2:shared.add(m)
        cancel=eq(r'\int_e\vec F\cdot d\vec r+\int_{-e}\vec F\cdot d\vec r=0',size=35,width=11).to_edge(DOWN,buff=.5)
        self.add_fixed_in_frame_mobjects(cancel)
        self.play(FadeIn(patch_loops),Write(cancel),run_time=2);self.wait(2)
        self.play(shared.animate.set_opacity(0),run_time=1.5);self.wait(1)
        self.play(FadeOut(patch_loops),FadeOut(cancel),run_time=1)
        currents=VGroup();inner=VGroup();outer=VGroup()
        for incidents in edge_incidence(faces).values():
            for a,b,fi in incidents:
                center=np.mean([xy[v] for v in faces[fi]],axis=0)
                m=surface_edge(xy[a],xy[b],1.4,MAROON_C if len(incidents)==2 else TEAL,center,width=1)
                currents.add(m);(inner if len(incidents)==2 else outer).add(m)
        self.play(FadeIn(currents),run_time=2);self.wait(2)
        self.play(LaggedStart(*[FadeOut(m) for m in inner],lag_ratio=.002),run_time=3)
        theorem=eq(r'\oint_{\partial S}\vec F\cdot d\vec r',r'=',r'\iint_S(\nabla\times\vec F)\cdot\hat n\,dS',size=41,width=11.8).to_edge(DOWN,buff=.55)
        theorem[0].set_color(TEAL);theorem[2].set_color(PURPLE)
        box=framed(theorem)
        self.add_fixed_in_frame_mobjects(theorem,box)
        self.play(Write(theorem),Create(box),run_time=2.5)
        newtitle=caption(r"Teorema di Stokes",size=32,color=WHITE).move_to(title)
        self.add_fixed_in_frame_mobjects(newtitle)
        self.play(FadeOut(title),FadeIn(newtitle),run_time=1)
        self.begin_ambient_camera_rotation(rate=.035)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(VGroup(mesh,outer)),run_time=1.5);self.wait(2)
        self.play(FadeOut(VGroup(curve,theorem,box,newtitle)),run_time=1.2)


class FundamentalTheorem(Scene):
    """Exact telescoping on an oriented interval, then the FTC."""
    def construct(self):
        title=caption(r"In una dimensione, il bordo sono due punti",size=30).move_to([0,3.15,0])
        xs=[-5.1,-1.7,1.7,5.1]; y=1.25
        dots=VGroup(*[Dot([x,y,0],color=YELLOW if i in (0,3) else WHITE,radius=.055) for i,x in enumerate(xs)])
        segments=VGroup(*[arrow([a+.08,y,0],[b-.08,y,0],TEAL,3,.16) for a,b in zip(xs[:-1],xs[1:])])
        labels=VGroup(*[eq(s,size=30).move_to([x,y+.45,0]) for s,x in zip(['a','x_1','x_2','b'],xs)])
        self.play(Write(title),Create(segments),FadeIn(dots),Write(labels),run_time=2);self.wait(1.5)
        parts=[]
        for i,(lo,hi) in enumerate([('a','x_1'),('x_1','x_2'),('x_2','b')]):
            q=eq(fr'f({hi})',r'-',fr'f({lo})',size=34,width=3.2).move_to([(xs[i]+xs[i+1])/2,-.05,0])
            parts.append(q)
        parts[0][2].set_color(YELLOW);parts[2][0].set_color(YELLOW)
        self.play(LaggedStart(*[Write(p) for p in parts],lag_ratio=.35),run_time=3)
        self.wait(1)
        pairs=[(parts[0][0],VGroup(parts[1][1],parts[1][2])),(parts[1][0],VGroup(parts[2][1],parts[2][2]))]
        for pair in pairs:
            self.play(*[m.animate.set_color(MAROON_C) for m in pair],run_time=.5)
            strokes=VGroup(*[Line(m.get_corner(DOWN+LEFT),m.get_corner(UP+RIGHT),color=MAROON_C,stroke_width=2) for m in pair])
            self.play(Create(strokes),run_time=.7);self.wait(.7)
            self.play(FadeOut(strokes),*[m.animate.set_opacity(0) for m in pair],run_time=.7)
        telescoping=eq(r'\sum_{i=0}^{N-1}\bigl[f(x_{i+1})-f(x_i)\bigr]',r'=',r'f(b)-f(a)',size=39,width=11.8).move_to([0,-1.6,0]);telescoping[2].set_color(YELLOW)
        self.play(Write(telescoping),run_time=2);self.wait(1.5)
        self.play(FadeOut(VGroup(*parts)),FadeOut(dots[1:3]),FadeOut(labels[1:3]),run_time=1)
        derivative=eq(r'f(x_{i+1})-f(x_i)',r'\approx',r'f\prime(x_i)\,\Delta x_i',size=36,width=10).move_to([0,-.05,0]);derivative[-1].set_color(PURPLE)
        self.play(Write(derivative),run_time=1.5);self.wait(2)
        integral=eq(r'\int_a^b f\prime(x)\,dx',r'=',r'f(b)-f(a)',size=49,width=10).move_to([0,-1.65,0]);integral[0].set_color(PURPLE);integral[2].set_color(YELLOW)
        self.play(ReplacementTransform(telescoping,integral),FadeOut(derivative),run_time=2)
        self.play(Create(framed(integral)));self.wait(3)
        self.play(FadeOut(Group(*self.mobjects)),run_time=1.2)


class GeneralizedStokes(Scene):
    """An economical visual synthesis in the existing mathematical language."""
    def construct(self):
        title=caption(r"Una sola struttura, in dimensioni diverse",size=31).move_to([0,3.2,0])
        centers=[np.array([-4.45,.65,0]),np.array([0,.65,0]),np.array([4.45,.65,0])]
        interval=VGroup(arrow(centers[0]+LEFT*1.1,centers[0]+RIGHT*1.1,TEAL,3,.13),
            Dot(centers[0]+LEFT*1.1,color=YELLOW),Dot(centers[0]+RIGHT*1.1,color=YELLOW))
        flat=ParametricFunction(lambda t:.53*curve_point(t)+centers[1],t_range=[0,2*PI,.03],color=TEAL,stroke_width=3)
        fill=Polygon(*[.53*curve_point(t)+centers[1] for t in np.linspace(0,2*PI,100,endpoint=False)],fill_color=TEAL_E,fill_opacity=.2,stroke_width=0)
        # Fixed orthographic projection for the 3D reminder; surface coordinates
        # are generated from the same mesh as StokesSurface, not a stock icon.
        project=lambda p:np.array([.54*(.85*p[0]-.5*p[1]),.54*(.36*p[0]+.61*p[1]+p[2]),0.])+centers[2]
        tiles,xy,faces,pos=surface_mesh(1.4,3,16)
        surf=VGroup(*[Polygon(*[project(pos(v)) for v in face],stroke_width=.6,stroke_color=TEAL_D,fill_color=TEAL_E,fill_opacity=.23) for face in faces])
        rim=ParametricFunction(lambda t:project(np.r_[curve_point(t)[:2],surface_z(*curve_point(t)[:2],1.4)]),t_range=[0,2*PI,.03],color=TEAL,stroke_width=3)
        diagrams=VGroup(interval,VGroup(fill,flat),VGroup(surf,rim))
        labels=VGroup(*[eq(s,size=31).move_to(c+UP*1.5) for s,c in zip(['1D','2D','3D'],centers)])
        formulas=VGroup(
            eq(r'\int_a^b f\prime\,dx=f(b)-f(a)',size=28,width=3.9),
            eq(r'\iint_D(Q_x-P_y)\,dA=\oint_{\partial D}\vec F\cdot d\vec r',size=27,width=3.9),
            eq(r'\iint_S(\nabla\times\vec F)\cdot\hat n\,dS=\oint_{\partial S}\vec F\cdot d\vec r',size=27,width=3.9))
        for f,c in zip(formulas,centers):f.move_to([c[0],-1.1,0])
        self.play(Write(title),run_time=1)
        for d,l,f in zip(diagrams,labels,formulas):
            self.play(FadeIn(d),Write(l),Write(f),run_time=2);self.wait(.7)
        self.wait(2)
        self.play(VGroup(diagrams,labels).animate.scale(.7).shift(UP*.5),FadeOut(formulas),FadeOut(title),run_time=2)
        main=eq(r'\int_M',r'd\omega',r'=',r'\int_{\partial M}',r'\omega',size=70,width=10.5).move_to([0,-.7,0])
        main[1].set_color(PURPLE);main[3].set_color(TEAL);main[4].set_color(TEAL)
        self.play(Write(main),run_time=2.5)
        definition=eq(r'\dim M=k,\qquad\omega\in\Omega^{k-1}(M)',size=31).move_to([0,-2.15,0])
        self.play(Write(definition),run_time=1.5);self.wait(2)
        words=caption(r"Il cambiamento nell'interno. La quantit\`a sul bordo.",size=30,color=WHITE).move_to([0,-2.85,0])
        self.play(Write(words),run_time=2);self.wait(2)
        self.play(FadeOut(VGroup(diagrams,labels,definition,words)),main.animate.move_to(ORIGIN),run_time=2)
        name=caption(r"Teorema di Stokes",size=37,color=WHITE).move_to([0,1.65,0])
        self.play(Write(name),Create(framed(main)),run_time=1.5);self.wait(4)
        self.play(FadeOut(Group(*self.mobjects)),run_time=1.5);self.wait(.5)
