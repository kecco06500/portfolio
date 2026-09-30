"""Circuitazione, dimostrazione sul quadrato e cancellazione corretta."""
from manim import *
from numpy import *
from math import *

from .stile import (eq, caption, framed, curve_point, curve_tangent, quadratic_field, backdrop, soft_field, arrow, domain_cells, edge_incidence, planar_mesh, surface_z, surface_normal, surface_topology, surface_mesh, lifted_curve, surface_edge, lowpoly_arrow)

class SumDotProducts(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.6,

            zoomed_display_width=config.frame_width / 3.5,
            zoomed_display_height=config.frame_height / 3.5,

            **kwargs
        )

    def construct(self):
        plane = NumberPlane().set_opacity(0.3)

        R = 2
        A = 0.2
        n = 5

        def curve_func(t):
            r = R + A * np.sin(n * t)
            x = r * np.cos(t)
            y = r * np.sin(t)
            return np.array([x, y, 0])

        curve = ParametricFunction(
            curve_func,
            t_range=[0, 2 * PI],
            color=TEAL,
            stroke_width=3
        )

        def color_func(p):
            return np.linalg.norm(p)

        def vector_field(p):
            x, y, _ = p
            return np.array([-y ** 2, x ** 2, 0])

        field = ArrowVectorField(
            vector_field,
            x_range=[-7, 7],
            y_range=[-4, 4],
            color_scheme=color_func,
            min_color_scheme_value=0,
            max_color_scheme_value=3,
            colors=[BLUE, GREEN, YELLOW, RED],
            opacity=0.4
        )

        def tangent_curve(t):
            r = R + A * np.sin(n * t)
            r_prime = A * n * np.cos(n * t)
            x_prime = r_prime * np.cos(t) - r * np.sin(t)
            y_prime = r_prime * np.sin(t) + r * np.cos(t)
            return np.array([x_prime, y_prime, 0])

        t = ValueTracker(0)
        punto = always_redraw(
            lambda: Dot(
                point=plane.c2p(*curve_func(t.get_value())[:2]),
                color=YELLOW,
                radius=0.05 * (plane.x_axis.unit_size + plane.y_axis.unit_size) / 2
            )
        )

        tangent_vector = always_redraw(lambda: Arrow(
            start=plane.c2p(*curve_func(t.get_value())[:2]),
            end=plane.c2p(*(curve_func(t.get_value())[:2] + 0.4 * tangent_curve(t.get_value())[:2])),
            buff=0,
            color=TEAL,
            max_tip_length_to_length_ratio=0.15
        ))

        field_vector = always_redraw(lambda: Arrow(
            start=plane.c2p(*curve_func(t.get_value())[:2]),
            end=plane.c2p(*(curve_func(t.get_value())[:2] + 0.4 * vector_field(curve_func(t.get_value()))[:2])),
            buff=0,
            max_tip_length_to_length_ratio=0.1,
            color=MAROON_C
        ))

        # Add all to the main scene (remains fullscreen)
        self.add(plane, curve, field, tangent_vector, field_vector, punto)

        self.wait()

        def follow_dot(m):
            m.move_to(punto.get_center())

        all_objects = VGroup(field, plane, curve, tangent_vector, field_vector)
        self.play(all_objects.animate.scale(0.5).shift(LEFT * 3))

        # No manual move_to before activating zooming!
        orig_frame = self.camera.frame.copy()

        # Activate zooming without animation, then animate frame to dot
        self.activate_zooming(animate=False)
        self.zoomed_display.to_corner(UR, buff=1)
        self.zoomed_camera.frame.save_state()
        self.play(
            self.zoomed_camera.frame.animate.move_to(punto.get_center()),
            run_time=1.2,
            rate_func=smooth
        )
        # Add updater only after initial animation for smooth tracking
        self.zoomed_camera.frame.add_updater(follow_dot)

        self.wait(2)

        ########

        side_tang_vec = always_redraw(lambda: Arrow(
            start=np.array([0, 0, 0]),
            end=0.3 * tangent_curve(t.get_value()),
            buff=0,
            max_tip_length_to_length_ratio=0.15,
            color=TEAL
        ).move_to(ORIGIN + RIGHT * 2 + DOWN * 1))

        dot_point = MathTex(r'\cdot').next_to(side_tang_vec, RIGHT, buff=0.5)

        side_dot_vec = always_redraw(lambda: Arrow(
            start=np.array([0, 0, 0]),
            end=0.3 * vector_field(curve_func(t.get_value())),
            buff=0,
            max_tip_length_to_length_ratio=0.15,
            color=MAROON_C
        ).next_to(dot_point, RIGHT, buff=0.5))

        equal_sign = MathTex(r'=').next_to(side_dot_vec, RIGHT, buff=1)

        # Value always redraw
        value_tex = DecimalNumber(0, num_decimal_places=2).scale(0.8)
        def update_value(m):
            value = np.dot(vector_field(curve_func(t.get_value())), tangent_curve(t.get_value()))
            m.set_value(value).set_color(GREEN if value > 0 else RED if value < 0 else WHITE)
            m.next_to(equal_sign, RIGHT, buff=0.5)
        value_tex.add_updater(update_value)
        update_value(value_tex)

        self.play(GrowArrow(side_tang_vec), GrowArrow(side_dot_vec), Create(dot_point), Create(equal_sign), Write(value_tex))

        self.play(t.animate.set_value(2 * pi), run_time=10)

        self.wait()
        self.zoomed_camera.frame.remove_updater(follow_dot)

        self.play(FadeOut(self.zoomed_camera.frame), run_time=0.6)
        value_tex.clear_updaters()

        self.play(FadeOut(side_dot_vec), FadeOut(side_tang_vec), FadeOut(dot_point), FadeOut(equal_sign), Unwrite(value_tex))

        self.play(FadeOut(self.zoomed_display), run_time=1.5, rate_func=smooth)

        self.wait()

        self.play(all_objects.animate.shift(RIGHT * 3).scale(2))

        self.wait()


class GreenAnalytic(Scene):
    def construct(self):


        quad1 = Square().shift(LEFT*4.1,DOWN*0.8)
        vertices = quad1.get_vertices()

        A = MathTex('B').next_to(quad1.get_vertices()[3], buff =0).shift(DOWN*0.3)
        B = MathTex('C').next_to(quad1.get_vertices()[0],buff= 0).shift(UP*0.3)
        C = MathTex('D').next_to(quad1.get_vertices()[1],buff =0 ).shift(LEFT*0.45).shift(UP*0.3)
        D = MathTex('A').next_to(quad1.get_vertices()[2],buff = 0).shift(LEFT*0.45).shift(DOWN*0.3)


        sides = [
            Line(vertices[i], vertices[(i+1) % 4], color=WHITE)
            for i in range(4)
        ]

        x_tips = [0,2]
        delta_xs = VGroup()
        x_labels = VGroup()
        for tip in x_tips:
            delta_x = Arrow(
                start = sides[tip].get_start(),
                end = sides[tip].get_end(),
                max_tip_length_to_length_ratio=0.15
            )
            delta_x.shift(UP*0.25) if tip == 0 else delta_x.shift(DOWN *0.25)
            delta_x_label = MathTex(r'\Delta x').scale(0.7).next_to(delta_x,UP) if tip==0 else MathTex(r'\Delta x').scale(0.7).next_to(delta_x,DOWN)
            delta_xs.add(delta_x)
            x_labels.add(delta_x_label)




        y_tips = [1,3]
        delta_ys = VGroup()
        y_labels = VGroup()
        for tip in y_tips:
            delta_y = Arrow(
                start = sides[tip].get_start(),
                end = sides[tip].get_end(),
                max_tip_length_to_length_ratio=0.15
            )
            delta_y.shift(LEFT*0.25) if tip == 1 else delta_y.shift(RIGHT *0.25)
            delta_y_label = MathTex(r'\Delta y').scale(0.7).next_to(delta_y,LEFT) if tip==1 else MathTex(r'\Delta y').scale(0.7).next_to(delta_y,RIGHT)
            delta_ys.add(delta_y)
            y_labels.add(delta_y_label)


        def vector_field(p):
            x, y, _ = p
            return np.array([-y,x,0])

        def color_func(p):
            return np.linalg.norm(p) -0.5

        field = ArrowVectorField(vector_field,
                                x_range= [-3,3],
                                y_range = [-3,3],
                                color_scheme= color_func,
                                min_color_scheme_value= 0,
                                max_color_scheme_value= 3,
                                colors = [BLUE,GREEN,YELLOW,RED],
                                opacity = 0.4).scale(0.7).shift(LEFT*4.1,DOWN*0.8)

        ev= VGroup(field,A,B,C,D,delta_xs,delta_ys,x_labels,y_labels)

        #Parte1

        formula00 = MathTex(r'\oint_{\partial R} \vec{F} \cdot d\vec{r} = \;').scale(0.6).to_corner(UL).shift(RIGHT*1.75,DOWN*0.5)
        formula01 = MathTex(r'\int_{A}^{B} P(x,y_A)dx ').scale(0.6).next_to(formula00,RIGHT,buff= 0.2)
        formula02 = MathTex(r'+ \int_{B}^{C}Q(x_B,y)dy' ).scale(0.6).next_to(formula01,RIGHT,buff= 0.2)
        formula03 = MathTex(r'+\int_{C}^{D}P(x,y_C)dx ').scale(0.6).next_to(formula02,RIGHT,buff= 0.2)
        formula04 = MathTex(r'+\int_{D}^{A}Q(x_D,y)dy ').scale(0.6).next_to(formula03,RIGHT,buff= 0.2)

        formulas_0 = VGroup(formula00,formula01,formula02,formula03,formula04)

        formula10 = MathTex(r'\oint_{\partial R} \vec{F} \cdot d\vec{r} =').scale(0.6).to_corner(UL).shift(DOWN *0.5)
        formula11 = MathTex(r'\int_x^{x+\Delta x} P(t,y)dt').scale(0.6).next_to(formula10,RIGHT,buff= 0.2)
        formula12 = MathTex(r'-\int_x^{x+\Delta x}P(t,y + \Delta y) dt').scale(0.6).next_to(formula11,RIGHT,buff= 0.2)
        formula13 = MathTex(r'+\int_y^{y+\Delta y} Q(x + \Delta x,t)dt').scale(0.6).next_to(formula12,RIGHT,buff= 0.2)
        formula14 = MathTex(r'- \int_y^{y+\Delta y} Q(x,t)dt =').scale(0.6).next_to(formula13,RIGHT,buff= 0.2)

        before1 = VGroup(formula14,formula13)
        before2 = VGroup(formula11,formula12)
        whole1 = VGroup(formula11,formula12,formula13,formula14)

        formula1_star1= MathTex(r'\int_y^{y+\Delta y} (Q(x +\Delta x,t) - Q(x,t))dt ').scale(0.6).next_to(formula10,RIGHT,buff=0.2)
        formula1_star2 =MathTex(r'+ \int_x^{x+\Delta x} (P(t,y) - P(t,y+\Delta y))dt').scale(0.6).next_to(formula1_star1,RIGHT,buff=0.2)

        up_whole1 = VGroup(formula10,formula1_star1,formula1_star2)

        Q_partial1 = MathTex(r'\frac{\partial Q}{\partial x} \approx \frac{Q(x +\Delta x,y) - Q(x,y) }{\Delta x}').scale(0.6).to_corner(UL).shift(RIGHT*1,DOWN*0.5)
        Q_partial2 = MathTex(r'\implies Q(x +\Delta x,y) - Q(x,y)\approx \frac{\partial Q}{\partial x} \Delta x').scale(0.6).next_to(Q_partial1)
        Q_partials = VGroup(Q_partial1,Q_partial2)

        P_partial1 = MathTex(r'\frac{\partial P}{\partial y} \approx \frac{P(x,y+\Delta y) -P(x,y)}{\Delta y}').scale(0.6).to_corner(UL).shift(RIGHT*1,DOWN*0.5)
        P_partial2 = MathTex(r'\implies P(x,y+\Delta y) - P(x,y) \approx \frac{\partial P}{\partial y}\Delta y').scale(0.6).next_to(Q_partial1)
        P_partials= VGroup(P_partial1,P_partial2)

        rec= Rectangle(height=5,width= 8, stroke_opacity =0.6).to_edge(RIGHT).shift(DOWN*1).set_color([TEAL_B,ORANGE])

        #Animazioni

        self.wait()
        self.play(Write(field))
        self.play(Create(quad1))
        self.play(Write(A),Write(B),Write(C),Write(D))
        self.wait()

        self.play(Create(rec))

        self.play(Write(formula00),Indicate(quad1))
        self.wait()
        self.play(Write(formula01),Indicate(sides[2]))
        self.wait()
        self.play(Write(formula02),Indicate(sides[3]))
        self.wait()
        self.play(Write(formula03),Indicate(sides[0]))
        self.wait()
        self.play(Write(formula04),Indicate(sides[1]))
        self.wait()

        self.remove(sides[1],sides[2],sides[3],sides[0])

        self.play(formulas_0.animate.scale(0.66).shift(RIGHT*2 ,DOWN*1.7))

        self.play(Write(formula10))
        self.play(Write(whole1))
        self.wait()
        self.play(TransformMatchingShapes(before1,formula1_star1), TransformMatchingShapes(before2,formula1_star2))
        self.play(*[GrowArrow(arr) for arr in delta_xs] , *[Write(label) for label in x_labels])
        self.play(*[GrowArrow(arr) for arr in delta_ys] , *[Write(label) for label in y_labels])
        self.wait()

        self.play(up_whole1.animate.scale(0.66).next_to(formulas_0,DOWN))
        self.wait()


        ##Parte 2

        formula20 = MathTex(r'\oint_{\partial R} \vec{F} \cdot d\vec{r} \approx').scale(0.4).next_to(up_whole1,DOWN).shift(LEFT*2)
        formula21 = MathTex(r'\int_y^{y +\Delta y}\frac{\partial Q}{\partial x} (x,t)\Delta x d t').scale(0.4).next_to(formula20,RIGHT,buff = 0.2)
        formula22 = MathTex(r'-\int_x^{x+\Delta x}\frac{\partial P}{\partial y} (t,y)\Delta y dt ').scale(0.4).next_to(formula21,RIGHT,buff = 0.2)

        whole2 = VGroup(formula20,formula21,formula22)

        formula2_star1 = MathTex(r'\int_y^{y +\Delta y}\frac{\partial Q}{\partial x} (x,t)\Delta x d t \approx \frac{\partial Q}{\partial x}\Delta x \Delta y').scale(0.6).to_corner(UL).shift(DOWN*0.5)
        formula2_star2 = MathTex(r'\int_x^{x +\Delta x}\frac{\partial P}{\partial y}(t,y)\Delta y dt \approx\frac{\partial P}{\partial y}\Delta x \Delta y').scale(0.6).to_corner(UR).shift(DOWN*0.5)

        formula30 = MathTex(r'\oint_{\partial R} \vec{F} \cdot d\vec{r} \approx').scale(0.4).next_to(formula20,DOWN)
        formula31 = MathTex(r'\frac{\partial Q}{\partial x}\Delta x \Delta y').scale(0.4).next_to(formula30,RIGHT,buff= 0.2)
        formula32 = MathTex(r'-\frac{\partial P}{\partial y} \Delta x \Delta y ').scale(0.4).next_to(formula31,RIGHT,buff= 0.2)
        formula33 = MathTex(r'= \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) \Delta x \Delta y').scale(0.5).next_to(formula32,RIGHT,buff= 0.2)

        formulas_3 = VGroup(formula30,formula31,formula32,formula33)

        final_formula = MathTex(r'\oint_{\partial R} \vec{F} \cdot d\vec{r} \approx \left( \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} \right) \Delta x \Delta y').to_edge(RIGHT)

        formula1_star1_copy = formula1_star1.copy()
        formula1_star2_copy = formula1_star2.copy()

        all_formulas = VGroup(formulas_0, up_whole1,whole2,formulas_3)

        rec2 = SurroundingRectangle(final_formula).set_color([TEAL_B,ORANGE])

        self.play(Write(Q_partial1))
        self.play(Write(Q_partial2))

        self.play(Write(formula20))
        self.play(ReplacementTransform(Q_partials,formula21),ReplacementTransform(formula1_star1_copy,formula21))

        self.play(Write(P_partial1))
        self.play(Write(P_partial2))

        self.play(ReplacementTransform(P_partials,formula22),ReplacementTransform(formula1_star2_copy,formula22))
        self.wait()

        self.play(Write(formula2_star1), Write(formula2_star2))
        self.wait()

        self.play(Write(formula30))
        self.wait()

        self.play(ReplacementTransform(formula2_star1, formula31),ReplacementTransform(formula21.copy(), formula31))
        self.play(ReplacementTransform(formula2_star2, formula32),ReplacementTransform(formula22.copy(), formula32))
        self.wait()
        self.play(Write(formula33))

        self.wait(3)

        self.play(ReplacementTransform(all_formulas,final_formula), ev.animate.shift(UP*0.8).scale(1.1), quad1.animate.shift(UP*0.8).scale(1.1), ReplacementTransform(rec,rec2))
        self.wait()


class TwoSquares(Scene):
    """Optional replacement of the existing incorrect shared-edge cancellation."""
    def construct(self):
        centers=[np.array([-1.05,.8,0]),np.array([1.05,.8,0])]
        sides=[];squares=[]
        for c in centers:
            vertices=[c+[-1.05,-1.05,0],c+[1.05,-1.05,0],c+[1.05,1.05,0],c+[-1.05,1.05,0]]
            squares.append(Polygon(*vertices,color=WHITE,stroke_width=2,fill_color=TEAL_E,fill_opacity=.08))
            sides.append(VGroup(*[arrow(.89*a+.11*c,.89*b+.11*c,WHITE,2,.13)
                for a,b in zip(vertices,vertices[1:]+vertices[:1])]))
        labels=VGroup(eq('R_1',size=32).move_to(centers[0]),eq('R_2',size=32).move_to(centers[1]))
        self.play(Create(squares[0]),LaggedStart(*[GrowArrow(a) for a in sides[0]],lag_ratio=.15),Write(labels[0]),run_time=2)
        self.play(Create(squares[1]),LaggedStart(*[GrowArrow(a) for a in sides[1]],lag_ratio=.15),Write(labels[1]),run_time=2)
        self.wait(1)
        shared1,shared2=sides[0][1],sides[1][3]
        self.play(shared1.animate.set_color(MAROON_C),shared2.animate.set_color(TEAL),run_time=1)
        terms=eq(r'+\int_y^{y+\Delta y}Q(x+\Delta x,t)\,dt',
                 r'-\int_y^{y+\Delta y}Q(x+\Delta x,t)\,dt',size=34,width=11.7).move_to([0,-1.45,0])
        terms[0].set_color(MAROON_C);terms[1].set_color(TEAL)
        self.play(Write(terms),run_time=2);self.wait(2)
        crosses=VGroup(*[Line(t.get_corner(DOWN+LEFT),t.get_corner(UP+RIGHT),color=t.get_color(),stroke_width=2) for t in terms])
        self.play(Create(crosses));self.wait(.7)
        zero=eq('0',size=40).move_to(terms)
        self.play(FadeOut(crosses),ReplacementTransform(terms,zero),FadeOut(shared1),FadeOut(shared2),run_time=1.5)
        outer=Polygon([-2.1,-.25,0],[2.1,-.25,0],[2.1,1.85,0],[-2.1,1.85,0],color=TEAL,stroke_width=3)
        self.play(FadeOut(VGroup(*squares)),Create(outer),run_time=1)
        identity=eq(r'\oint_{\partial R_1}\vec F\cdot d\vec r',r'+',r'\oint_{\partial R_2}\vec F\cdot d\vec r',r'=',r'\oint_{\partial(R_1\cup R_2)}\vec F\cdot d\vec r',size=37,width=12).move_to([0,-2.45,0])
        identity[-1].set_color(TEAL)
        self.play(FadeOut(zero),Write(identity),run_time=2);self.play(Create(framed(identity)));self.wait(3)
