"""Campo, rotore e ipotesi. Conserva le modifiche locali ai flussi."""
from manim import *
from numpy import *
from math import *

class VecFieldWithSteam(Scene):
    def construct(self):
        plane = NumberPlane(y_range = [-2*PI, 2*PI]).set_opacity(0.3)

        def color_func(p):
            plane_coords = plane.c2p(p)
            return np.linalg.norm(plane_coords)

        def vector_field(p):
            x, y,_= p
            return np.array([np.sin(y) - 0.5*x, np.cos(x) + 0.5*y, 0])

        field = ArrowVectorField(vector_field,
                                x_range= [-7,7],
                                y_range= [-6,6],
                                color_scheme=color_func,
                                min_color_scheme_value=0,
                                max_color_scheme_value=2,
                                colors=[BLUE,GREEN,YELLOW,RED])

        self.play(Write(plane),Write(field))

        stream_lines = StreamLines(
            vector_field,
            x_range=[-2*PI,2*PI],
            y_range=[-PI,PI],
            stroke_width=1.5,
            max_anchors_per_line=30,
            color_scheme=color_func,
            min_color_scheme_value=0,
            max_color_scheme_value=2,
            colors=[BLUE,GREEN,YELLOW,RED]
        )

        self.add(stream_lines)

        stream_lines.start_animation(
            flow_speed=1,
            time_width= 0.5,
        )

        self.wait(8)
        stream_lines.end_animation()

        self.wait()


        highlight = SurroundingRectangle(VGroup(plane,field,stream_lines),color = YELLOW, buff = 0)
        self.add(highlight)
        # Group and move only static objects (plane + field)
        static_objs = VGroup(plane, field, highlight)
        static_objs.generate_target()
        static_objs.target.scale(0.4)
        static_objs.target.shift(LEFT*3.5,DOWN *0.7)

        stream_lines.generate_target()
        stream_lines.target.scale(0.4)
        stream_lines.target.shift(LEFT*3.5,DOWN * 0.7)

        self.play(
            MoveToTarget(static_objs),
            MoveToTarget(stream_lines)
            )

        # Add formulas on the right

        formule = []

        formula1 = MathTex(r'\mathbb{F}(x,y) =\begin{bmatrix} P (x,y) \\ Q(x,y) \end{bmatrix}= \begin{bmatrix} \sin(y) - 0.5x \\ \cos(x) + 0.5y \end{bmatrix}').scale(0.6).to_edge(UP,buff= 0.7).shift(LEFT*4)
        formula2 = MathTex(r'(\nabla \times \mathbb{F} ) \cdot \hat{z} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}').scale(0.6).to_edge(UP,buff= 0.7).shift(RIGHT*4)
        formula3 = MathTex(r'P(x,y) = \sin(y) -0.5x \implies \frac{\partial P}{\partial y} = \cos(y)').scale(0.6).next_to(formula2, DOWN, buff = 1)
        formula4 = MathTex(r'Q(x,y) = \cos(x) +0.5y \implies \frac{\partial Q}{\partial x} = -\sin(x)').scale(0.6).next_to(formula3, DOWN)
        formula5 = MathTex(r'(\nabla \times \mathbb{F}) \cdot \hat{z} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = -\sin(x) - \cos(y)').scale(0.6).next_to(formula4,DOWN)
        formula6 = MathTex(r'(\nabla \times \mathbb{F}) \cdot \hat{z}> 0 \iff \sin(x)+\cos(y)<0 ').scale(0.6).next_to(formula5,DOWN)

        formula1[0][0].set_color(PURPLE)
        formula1[0][8:14].set_color(MAROON_C)
        formula1[0][14:20].set_color(TEAL_D)



        formula2[0][6:8].set_color(BLUE_D)
        formula2[0][0:5].set_color(PURPLE)
        formula2[0][9:14].set_color(TEAL_D)
        formula2[0][15:21].set_color(MAROON_C)



        formula3[0][0:6].set_color(MAROON_C)
        formula3[0][20:25].set_color(MAROON_C)


        formula4[0][:6].set_color(TEAL_D)
        formula4[0][20:25].set_color(TEAL_D)

        formula5[0][:5].set_color(PURPLE)
        formula5[0][6:8].set_color(BLUE_D)
        formula5[0][9:14].set_color(TEAL_D)
        formula5[0][15:20].set_color(MAROON_C)



        formula6[0][0:5].set_color(PURPLE)
        formula6[0][6:8].set_color(BLUE_D)


        formule.extend([formula1,formula2,formula3,formula4, formula5, formula6])

        for i in formule:
            self.play(Write(i))
            self.wait(1)

        self.wait()

        self.play(Unwrite(formula2), Unwrite(formula3), Unwrite(formula4), Unwrite(formula5))
        self.wait()

        self.play(formula6.animate.to_edge(UP,buff= 0.8))


        self.wait()


        plane2 = NumberPlane(
            x_range = [-7,7,1],
            y_range =[-2*PI,2*PI,1]).set_opacity(0.3).scale(0.4).shift(RIGHT *3.5, DOWN *0.7)



        axes = Axes(
            x_range = [-7,7],
            y_range = [-5,5],
            x_length= 14,
            y_length= 10
        ).scale(0.4).move_to(plane2.get_center())

        func1 = axes.plot(lambda x: np.arccos(-sin(x)), stroke_width = 1)
        func2 = axes.plot(lambda x: -np.arccos(-sin(x)), stroke_width = 1)
        func3= axes.plot(lambda x: -np.arccos(-sin(x)) + 2*PI,  stroke_width = 1)
        func4 = axes.plot(lambda x: np.arccos(-sin(x)) - 2*PI, stroke_width = 1 )


        area1 = axes.get_area(func1, bounded_graph= func3)
        area2 = axes.get_area(func2, bounded_graph= func4)

        self.play(Write(plane2))

        animations = [Create(func1), Create(func2), Create(func3), Create(func4)]
        self.play(*animations)

        self.wait()

        RightSide = VGroup(func1,func2,func3,func4, area1,area2,plane2)

        self.play(FadeIn(area1), FadeIn(area2))

        self.wait()

        self.play(Unwrite(formula1), Unwrite(formula6))

        static_objs.target.shift(RIGHT*3.5,UP*0.7)
        static_objs.target.scale(2.5)

        stream_lines.target.shift(RIGHT*3.5,UP*0.7)
        stream_lines.target.scale(2.5)

        RightSide.generate_target()
        RightSide.target.shift(LEFT*3.5, UP *0.7)
        RightSide.target.scale(2.5)



        self.play(MoveToTarget(static_objs),
                  MoveToTarget(stream_lines),
                  MoveToTarget(RightSide))

        self.play(Uncreate(highlight), Uncreate(plane2))


        self.wait()


class VecFieldWithSteamContinua(ThreeDScene):
    def construct(self):

        plane = NumberPlane(x_range= [-7,7],
                            y_range =[-2*pi,2*pi]).set_opacity(0.3)
        axes = ThreeDAxes(
            x_range = [-7,7],
            y_range = [-5,5],
            z_range = [-5,5],
            x_length= 14,
            y_length= 10
        ).move_to(plane.get_center())

        def color_func(p):
            x,y, _ = p
            return np.sqrt(x**2 + y**2)

        def vector_field(p):
            x, y,_= p
            return np.array([np.sin(y) - 0.5*x, np.cos(x) + 0.5*y, 0])

        def curl_vector(p):
            x,y,_ = p
            curl_z= -np.sin(x) -np.cos(y)
            return np.array([0,0,curl_z])

        sample_points = [plane.c2p(3*pi/2,pi),
                        plane.c2p(-pi/2,pi),
                        plane.c2p(-3*pi/2,0),
                        plane.c2p(pi/2,0),
                        plane.c2p(-pi/2,-pi),
                        plane.c2p(3*pi/2,-pi)]

        curl_arrows = VGroup()
        for p in sample_points:
            curl = curl_vector(p)
            colors = RED if curl[2] > 0 else BLUE
            arrow = Arrow3D(start = p,
                            end =p +curl,
                            color = colors,
                            resolution = 10)
            curl_arrows.add(arrow)

        field = ArrowVectorField(vector_field,
                                x_range= [-7,7],
                                y_range= [-6,6],
                                color_scheme=color_func,
                                min_color_scheme_value=0,
                                max_color_scheme_value=2,
                                colors=[BLUE,GREEN,YELLOW,RED])


        stream_lines=StreamLines(
            vector_field,
            x_range=[-2*PI,2*PI],
            y_range=[-PI,PI],
            stroke_width=1.5,
            max_anchors_per_line=30,
            color_scheme=color_func,
            min_color_scheme_value=0,
            max_color_scheme_value=2,
            colors=[BLUE,GREEN,YELLOW,RED]
        )

        self.add(plane,field,stream_lines)

        stream_lines.start_animation(
            flow_speed=1,
            time_width= 0.5,
        )

        self.wait(8)
        stream_lines.end_animation()

        func1 = axes.plot(lambda x: np.arccos(-sin(x)), stroke_width = 1)
        func2 = axes.plot(lambda x: -np.arccos(-sin(x)), stroke_width = 1)
        func3= axes.plot(lambda x: -np.arccos(-sin(x)) + 2*PI,  stroke_width = 1)
        func4 = axes.plot(lambda x: np.arccos(-sin(x)) - 2*PI, stroke_width = 1 )


        area1 = axes.get_area(func1, bounded_graph= func3)
        area2 = axes.get_area(func2, bounded_graph= func4)

        self.add(func1,func2,func3,func4,area1,area2)

        arc = Arc(radius=1, start_angle=0, angle=2*PI, color=GRAY)


        teal_arrows = []
        for i in range(3):
            for j in range(2):
                arrows1 = VGroup(
                CurvedArrow(arc.point_from_proportion(0.0), arc.point_from_proportion(0.33)),
                CurvedArrow(arc.point_from_proportion(0.33), arc.point_from_proportion(0.66)),
                CurvedArrow(arc.point_from_proportion(0.66), arc.point_from_proportion(0.0))
                ).move_to(plane.c2p(-PI/2 + i*2*PI,PI -j * 2*PI)).set_color(TEAL_E)
                teal_arrows.append(arrows1)

        red_arrows = []
        for i in range(2):
            arrows2 = VGroup(
                CurvedArrow(arc.point_from_proportion(0.0), arc.point_from_proportion(0.66), angle = -PI/2),
                CurvedArrow(arc.point_from_proportion(0.66), arc.point_from_proportion(0.33),angle = -PI/2),
                CurvedArrow(arc.point_from_proportion(0.33), arc.point_from_proportion(0.0),angle = -PI/2)
                ).move_to(plane.c2p(-3*PI/2 + i*2*PI,0)).set_color(RED_A)
            red_arrows.append(arrows2)

        animations = [FadeIn(arrow) for arrow in teal_arrows]
        self.play(animations)

        animations2 = [FadeIn(arrow) for arrow in red_arrows]
        self.play(animations2)


       # Rotate all teal arrows counter-clockwise
        animations = [
            Rotate(group, angle=2*PI, about_point=group.get_center(), rate_func =smooth, run_time=4)
            for group in teal_arrows
        ] + [
            Rotate(group, angle=-2*PI, about_point=group.get_center(), rate_func=smooth, run_time=4)
            for group in red_arrows
        ]

        self.play(*animations)
        self.wait(4)
        self.move_camera(phi = 60*DEGREES, theta = -45*DEGREES, distance= 65)


        self.wait()

        self.play(plane.animate.set_opacity(0.5))
        self.play(field.animate.set_opacity(0.5))


        self.play(FadeOut(stream_lines))

        animations2 = [FadeIn(curled_arrow) for curled_arrow in curl_arrows]

        self.play(*animations2)

        self.wait()

        self.begin_3dillusion_camera_rotation(rate= 2)

        self.wait(6)

        self.stop_3dillusion_camera_rotation()

        self.wait()


class Hypotesys(Scene):
    def construct(self):

        def vector_field1(p):
            x, y,_= p
            if x==0 and y==0:
                return np.array([0,0,0])
            return np.array([x*y/(x**2 + y**2),x*y, 0])

        def vector_field2(p):
            x,y,_ = p
            if x >= 0:
                return np.array([3*x, y, 0])
            elif x<0:
                return np.array([-3*x, y, 0])

        field1 = ArrowVectorField(vector_field1,
                                x_range= [-7,7],
                                y_range= [-6,6],
                                min_color_scheme_value=0,
                                max_color_scheme_value=5,
                                colors=[BLUE,GREEN,YELLOW,RED])

        stream_lines1 = StreamLines(
            vector_field1,
            x_range=[-2*PI,2*PI],
            y_range=[-PI,PI],
            stroke_width=1.5,
            max_anchors_per_line=30,
            min_color_scheme_value=0,
            max_color_scheme_value=5,
            colors=[BLUE,GREEN,YELLOW,RED]
        )

        field2 = ArrowVectorField(vector_field2,
                                x_range= [-7,7],
                                y_range= [-6,6],
                                min_color_scheme_value=0,
                                max_color_scheme_value=5,
                                colors=[BLUE,GREEN,YELLOW,RED])


        stream_lines2 = StreamLines(
            vector_field2,
            x_range=[-2*PI,2*PI],
            y_range=[-PI,PI],
            stroke_width=1.5,
            max_anchors_per_line=30,
            min_color_scheme_value=0,
            max_color_scheme_value=5,
            colors=[BLUE,GREEN,YELLOW,RED]
        )

        rec = Rectangle(fill_color = BLACK,
                        fill_opacity = 0.8).to_corner(UR)
        eq1 = MathTex(r'\mathbb{F} = \begin{bmatrix}\frac{xy}{x^2 +y^2}\\xy\end{bmatrix}').scale(0.8).move_to(rec.get_center())
        eq2 = MathTex(r'\mathbb{F} = \begin{cases}\begin{bmatrix}3x\\y\end{bmatrix} \;\;\;\; \;x\ge0\\\begin{bmatrix}-3x\\y\end{bmatrix}\;\;\; x<0\end{cases}').scale(0.6).move_to(rec.get_center())

        self.add(field1,rec)
        self.add(eq1)
        self.wait()

        self.add(stream_lines1)
        self.bring_to_front(rec)
        self.bring_to_front(eq1)

        stream_lines1.start_animation()

        self.wait(5)

        stream_lines1.end_animation()
        self.play(FadeOut(stream_lines1))


        self.play(ReplacementTransform(field1,field2), ReplacementTransform(eq1,eq2))

        self.add(stream_lines2)
        self.bring_to_front(rec)
        self.bring_to_front(eq2)


        stream_lines2.start_animation()

        self.wait(5)

        stream_lines2.end_animation()
