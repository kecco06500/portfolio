"""Prove storiche conservate. GreenVisual e Finale sono abbozzi superati da GreenGlobal.
Non fanno parte del rendering principale automatico.
"""
from manim import *
from numpy import *
from math import *

class VecFieldWithMass(Scene):
    def construct(self):

        plane = NumberPlane().set_opacity(0.3)
        self.add(plane)


        num_masses=2

        mass_positions = [np.array([i,j,0], dtype = float) for i in range(-num_masses, num_masses+1) for j in range(-num_masses,num_masses+1)]

        mass_velocities = [np.array([0,0,0], dtype=float) for _ in range(len(mass_positions))]

        masses = [Dot(point = pos, color = YELLOW) for pos in mass_positions]
        for m in masses:
            self.add(m)

        def color_func(p):
            return np.linalg.norm(p)

        def vector_field(p):
            x, y, _ = p
            return np.array([np.sin(y), np.sin(x), 0])

        field = ArrowVectorField(vector_field,
                                color_scheme= color_func,
                                min_color_scheme_value= 0,
                                max_color_scheme_value= 1.5,
                                colors = [BLUE,GREEN,RED])
        self.add(field)


        def divergence(p, h = 1e-2):
            x,y,_ = p
            Fx_x = (vector_field(np.array([x + h,y,0]))[0]- vector_field(np.array([x-h,y,0]))[0])
            Fy_y = (vector_field(np.array([x,y+h,0]))[1]- vector_field(np.array([x,y-h,0]))[1])
            return Fx_x +Fy_y


        def move_masses(i):
            def move_mass(m, dt):
                v = vector_field(mass_positions[i])
                mass_velocities[i] += v *dt
                mass_velocities[i] *= 0.99
                mass_positions[i] += mass_velocities[i] * dt


                if divergence(mass_positions[i]) < -0.2:
                    m.clear_updaters()

                m.move_to(mass_positions[i])
            return move_mass

        for i,j in enumerate(masses):
            j.add_updater(move_masses(i))
        self.wait(10)


class VecFieldWithSteam2(Scene):
    def construct(self):
        # Create the background plane
        plane = NumberPlane().set_opacity(0.3)
        self.add(plane)


        def color_func(p):
            return np.linalg.norm(p)
        # Define the vector field
        def vector_field(p):
            x, y, _ = p
            return np.array([-y,x,0])

        field = ArrowVectorField(vector_field,
                                color_scheme= color_func,
                                min_color_scheme_value= 0,
                                max_color_scheme_value= 2,
                                colors = [BLUE,GREEN,YELLOW,RED])
        self.add(field)

        # Create animated streamlines that follow the vector field
        stream_lines = StreamLines(
            vector_field,
            x_range=[-4, 4],
            y_range=[-3, 3],
            stroke_width=1.5,
            max_anchors_per_line=30,
            color_scheme= color_func,
            min_color_scheme_value= 0,
            max_color_scheme_value= 2,
            colors = [BLUE,GREEN,YELLOW,RED],
            opacity= 0.7)


        # Add streamlines to the scene
        self.add(stream_lines)


        # Animate the motion of the lines along the field
        stream_lines.start_animation(
            flow_speed=1.0,  # how fast they move
            time_width=0.5,  # how long each line is visible
        )

        self.wait(8)
        stream_lines.end_animation()


class GreenVisual(MovingCameraScene):
    def construct(self):


        plane = NumberPlane().set_opacity(0.3)

        R= 2
        A = 0.2
        n = 5


        def curve_func(t):
            r = R + A*np.sin(n*t)
            x = r *np.cos(t)
            y = r*np.sin(t)
            return np.array([x,y,0])

        curve = ParametricFunction(
            curve_func,
            t_range=[0, 2*PI],
            color= TEAL,
            stroke_width = 3
        )


        def color_func(p):
            return np.linalg.norm(p)

        def vector_field(p):
            x, y, _ = p
            return np.array([-y**2,x**2,0])

        field = ArrowVectorField(vector_field,
                                x_range= [-7,7],
                                y_range = [-4,4],
                                color_scheme= color_func,
                                min_color_scheme_value= 0,
                                max_color_scheme_value= 3,
                                colors = [BLUE,GREEN,YELLOW,RED],
                                opacity = 0.4)

        def tangent_curve(t):
            r = R + A * np.sin(n*t)
            r_prime = A * n * np.cos(n*t)
            x_prime = r_prime * np.cos(t) - r * np.sin(t)
            y_prime = r_prime * np.sin(t) + r * np.cos(t)
            return np.array([x_prime, y_prime, 0])


        t = ValueTracker(1)
        punto = always_redraw(
            lambda: Dot(
                point=plane.c2p(*curve_func(t.get_value())[:2]),
                color=YELLOW,
                radius=0.05 * (plane.x_axis.unit_size + plane.y_axis.unit_size) / 2
            )
        )

        tangent_vector = always_redraw(lambda: Arrow(
            start = plane.c2p(*curve_func(t.get_value())[:2]),
            end = plane.c2p(*(curve_func(t.get_value())[:2] + 0.4*tangent_curve(t.get_value())[:2])),
            buff = 0,
            color = TEAL,
            max_tip_length_to_length_ratio=0.15
        ))

        field_vector = always_redraw(lambda: Arrow(
            start = plane.c2p(*curve_func(t.get_value())[:2]),
            end = plane.c2p(*(curve_func(t.get_value())[:2] + 0.4*vector_field(curve_func(t.get_value()))[:2])),
            buff = 0,
            max_tip_length_to_length_ratio=0.1,
            color = MAROON_C
        ))


        self.add(plane,field,curve, tangent_vector, field_vector)

        self.wait()

        self.remove(tangent_vector,field_vector, punto, field)

        self.wait()

        inf = ValueTracker(1)
        x_len = 3*PI/2
        y_len = 3*PI/2

        grid = always_redraw(lambda: NumberPlane(
            x_range=[-inf.get_value(), inf.get_value()],
            y_range=[-inf.get_value(), inf.get_value()],
            x_length=x_len,
            y_length=y_len,
            background_line_style={"stroke_color": WHITE}
        ).set_opacity(0.6))



        self.add(grid)

        self.wait()

        self.play(inf.animate.set_value(3))

        def quad_arrows(center,scale):
            x,y,z = center
            s = scale/2 +0.05
            return VGroup(
                Arrow(start=[x-s, y-s, z], end=[x+s, y-s, z], buff=0).set_color(MAROON_C).set_opacity(0.5),
                Arrow(start=[x+s, y-s, z], end=[x+s, y+s, z], buff=0).set_color(MAROON_C).set_opacity(0.5),
                Arrow(start=[x+s, y+s, z], end=[x-s, y+s, z], buff=0).set_color(MAROON_C).set_opacity(0.5),
                Arrow(start=[x-s, y+s, z], end=[x-s, y-s, z], buff=0).set_color(MAROON_C).set_opacity(0.5),
            )

        dots_as_arrows = always_redraw(lambda: VGroup(*[
            quad_arrows(grid.c2p(x, y), scale=1/(inf.get_value()))
            for x in np.linspace(-1 + 1/(2*inf.get_value()), 1 - 1/(2*inf.get_value()), int(2*inf.get_value()))
            for y in np.linspace(-1 + 1/(2*inf.get_value()), 1 - 1/(2*inf.get_value()), int(2*inf.get_value()))
        ]))


        self.wait()

        self.add(dots_as_arrows)

        self.wait()

        self.play(self.camera.frame.animate.scale(0.3))

        self.play(inf.animate.set_value(6))


class Green_approx(MovingCameraScene):
    def construct(self):

        plane = NumberPlane(y_range = [-2*PI, 2*PI]).set_opacity(0.3)
        self.add(plane)

        def vector_field(p):
            x, y, _ = p
            return np.array([-y**2, x**2, 0])


        field = ArrowVectorField(
            vector_field,
            x_range=[-7, 7],
            y_range=[-5, 5],
            colors=[BLUE, GREEN, YELLOW, RED],
            opacity=0.4
        )

        self.add(field)
        self.wait()


        for n in range(3):
            stream = StreamLines(
                vector_field,
                x_range=[-6,6],
                y_range=[-5,5],
                stroke_width=1,
                max_anchors_per_line=30,
                n_repeats= 1
            )

            self.add(stream)
            stream.start_animation(flow_speed=1.2, time_width=0.5)
            self.wait(2)
            stream.end_animation()
            self.play(FadeOut(stream))
            self.play(self.camera.frame.animate.scale(0.5).move_to([0, 1, 0]))

        stream2 = StreamLines(
                vector_field,
                x_range=[-6,6],
                y_range=[-5,5],
                stroke_width=1,
                max_anchors_per_line=30,
                n_repeats= 5
            )
        self.add(stream2)
        stream2.start_animation(flow_speed=1.2, time_width=0.5)
        self.wait(2)
        stream2.end_animation()

        self.wait()


class ParialApprox(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.3,
            zoomed_display_height=2,
            zoomed_display_width=2,
            image_frame_stroke_width=40,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
            },
            **kwargs
        )

    def construct(self):

        eq1 = MathTex(r'\int_y^{y+\Delta y}\frac{\partial Q}{\partial x}(x,t)\Delta x dt \approx').to_edge(UP).scale(0.6).shift(LEFT*4.5,DOWN*2)
        eq2 = MathTex(r'\frac{\partial Q}{\partial x}(x_0,y_0)\Delta x \int_y^{y+\Delta y}dt =').next_to(eq1,RIGHT,buff =-0.9).scale(0.6)
        eq3 = MathTex(r'\frac{\partial Q}{\partial x}(x_0,y_0)\Delta x  \left( y +\Delta y -y\right) =').next_to(eq1,DOWN,buff =0.5).scale(0.6).shift(RIGHT*0.5)
        eq4 = MathTex(r'\frac{\partial Q}{\partial x}(x_0,y_0)\Delta x \Delta y  ').next_to(eq3,RIGHT,buff =-0.5).scale(0.6)


        plane = NumberPlane(
            x_range = [-5,5],
            y_range= [-5,5],
            x_length= 5,
            y_length= 5,
            background_line_style= {"stroke_opacity": 0.25}
        ).to_edge(RIGHT)

        def vector_field(p):
            x, y, _ = p
            return np.array([-y**2, x**2, 0])  # example field; you can change to desired vector field

        field = ArrowVectorField(
            vector_field,
            x_range=[-5,5],
            y_range=[-5,5],
            colors=[BLUE, GREEN, YELLOW, RED],
            min_color_scheme_value=0,
            max_color_scheme_value=5,
            opacity=0.6
        ).scale(0.5).to_edge(RIGHT)

        self.play(Write(plane), Write(field))


        self.wait(2)

        self.zoomed_camera.frame.move_to(plane.c2p(0,0))

        self.activate_zooming(animate=True)

        self.play(Write(eq1))

        self.play(self.zoomed_camera.frame.animate.move_to(plane.c2p(2,2)), Write(eq2))

        self.play(self.zoomed_camera.frame.animate.move_to(plane.c2p(-1,2)), Write(eq3))

        self.play(self.zoomed_camera.frame.animate.move_to(plane.c2p(-2,-2)), Write(eq4))


        self.wait()


class Finale(MovingCameraScene):
    def construct(self):

        plane = NumberPlane(
            x_range = [-5,5],
            y_range= [-5,5],
            x_length= 5,
            y_length= 5,
            background_line_style= {"stroke_opacity":0,
                                    "stroke_color": WHITE}
        )


        curve = ParametricFunction(lambda t: np.array([(1.2 + 0.2*sin(5*t))* cos(t), (1.2 +0.10*cos(3*t))*sin(t),0]),
                                   t_range= [0,2*pi])

        curve_label = MathTex(r'\gamma').next_to(curve, UR, buff =-0.2)

        graph = VGroup(plane,curve,curve_label)

        t= ValueTracker(1)
        grid = always_redraw(lambda :NumberPlane(
            x_range = [-t.get_value(),t.get_value()],
            y_range= [-t.get_value(),t.get_value()],
            x_length= 3,
            y_length= 3,
            background_line_style= {"stroke_opacity":0.7,
                                    "stroke_color": BLUE_E}).move_to(curve.get_center()))

        self.add(graph,grid)

        def quad_arrows(center,scale):
            x,y,z = center
            s = scale/2  -0.005
            return VGroup(
                Arrow(start=[x-s, y-s, z], end=[x+s, y-s, z], buff=0).set_color(WHITE).set_opacity(0.5),
                Arrow(start=[x+s, y-s, z], end=[x+s, y+s, z], buff=0).set_color(WHITE).set_opacity(0.5),
                Arrow(start=[x+s, y+s, z], end=[x-s, y+s, z], buff=0).set_color(WHITE).set_opacity(0.5),
                Arrow(start=[x-s, y+s, z], end=[x-s, y-s, z], buff=0).set_color(WHITE).set_opacity(0.5),
            )

        dots_as_arrows = always_redraw(lambda: VGroup(*[
            quad_arrows(grid.c2p(x, y), scale=1 / (t.get_value()))
            for i, x in enumerate(np.linspace(-1 + 1 / (2 * t.get_value()), 1 - 1 / (2 * t.get_value()), int(2 * t.get_value())))
            for j, y in enumerate(np.linspace(-1 + 1 / (2 * t.get_value()), 1 - 1 / (2 * t.get_value()), int(2 * t.get_value())))
            if 0 < i < int(2 * t.get_value()) - 1 and 0 < j < int(2 * t.get_value()) - 1
        ]))

        self.bring_to_front(curve)

        self.wait()

        self.play(t.animate.set_value(4))

        self.play(self.camera.frame.animate.scale(0.3).move_to(grid.get_center()))
