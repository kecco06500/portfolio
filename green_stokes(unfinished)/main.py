from manim import *
from numpy import *
from manim import PointCloudDot
from matplotlib.path import Path
from math import *
from random import *


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
            n_cycles = 1,
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
            n_cycles = 1,
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
            n_cycles=2       # number of loops through the field
        )

        self.wait(8)
        stream_lines.end_animation()


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
        value_tex = always_redraw(
            lambda: MathTex(
                f"{np.dot(vector_field(curve_func(t.get_value())), tangent_curve(t.get_value())):.2f}",
                color=GREEN if np.dot(vector_field(curve_func(t.get_value())), tangent_curve(t.get_value())) > 0
                else RED if np.dot(vector_field(curve_func(t.get_value())), tangent_curve(t.get_value())) < 0
                else WHITE
            ).scale(0.8).next_to(equal_sign, RIGHT, buff=0.5)
        )

        self.play(GrowArrow(side_tang_vec), GrowArrow(side_dot_vec), Create(dot_point), Create(equal_sign), Write(value_tex))

        self.play(t.animate.set_value(2 * pi), run_time=10)

        self.wait()
        self.zoomed_camera.frame.remove_updater(follow_dot)

        self.play(
            self.zoomed_camera.frame.animate.move_to(orig_frame.get_center()).scale(
                orig_frame.get_width() / self.zoomed_camera.frame.get_width()
            ),
            run_time=2,
            rate_func=smooth
        )

        self.play(FadeOut(side_dot_vec), FadeOut(side_tang_vec), FadeOut(dot_point), FadeOut(equal_sign), Unwrite(value_tex))

        self.play(FadeOut(self.zoomed_display), run_time=1.5, rate_func=smooth)

        self.wait()

        self.play(all_objects.animate.shift(RIGHT * 3).scale(2))

        self.wait()


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
        formula03 = MathTex(r'-\int_{C}^{D}P(x,y_C)dx ').scale(0.6).next_to(formula02,RIGHT,buff= 0.2)
        formula04 = MathTex(r'-\int_{D}^{A}Q(x_D,y)dy ').scale(0.6).next_to(formula03,RIGHT,buff= 0.2)
        
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
            stream.start_animation(flow_speed=1.2, time_width=0.5, n_cycles=1)
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
        stream2.start_animation(flow_speed=1.2, time_width=0.5, n_cycles=1)
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


class TwoSquares(Scene):
    def construct(self):
        
        quad1 = Square()
        
        label1= always_redraw(lambda:MathTex(r'\partial{R_1}').scale(0.7).next_to(quad1,DL,buff= 0.2).shift(RIGHT*0.4))
        
        def vector_field(p):
            x, y, _ = p
            return np.array([-y,x,0])
 
        field = ArrowVectorField(vector_field,
                                x_range= [-3,3],
                                y_range = [-3,3],
                                min_color_scheme_value= 0,
                                max_color_scheme_value= 3,
                                colors = [BLUE,GREEN,YELLOW,RED],
                                opacity = 0.4)


    
        def create_square_arrows(square, delta_shift=0.25, label_scale=0.7):
            vertices = square.get_vertices()
            sides = [Line(vertices[i], vertices[(i+1)%4], color=WHITE) for i in range(4)]
        
            # Δx arrows
            x_tips = [0,2]
            delta_xs = VGroup()
            x_labels = VGroup()
            for tip in x_tips:
                delta_x = Arrow(
                    start=sides[tip].get_start(),
                    end=sides[tip].get_end(),
                    max_tip_length_to_length_ratio=0.15
                )
                delta_x.shift(UP*delta_shift) if tip == 0 else delta_x.shift(DOWN*delta_shift)
                delta_x_label = MathTex(r'\Delta x').scale(label_scale).next_to(delta_x, UP) if tip == 0 else MathTex(r'\Delta x').scale(label_scale).next_to(delta_x, DOWN)
                delta_xs.add(delta_x)
                x_labels.add(delta_x_label)
        
            # Δy arrows
            y_tips = [1,3]
            delta_ys = VGroup()
            y_labels = VGroup()
            for tip in y_tips:
                delta_y = Arrow(
                    start=sides[tip].get_start(),
                    end=sides[tip].get_end(),
                    max_tip_length_to_length_ratio=0.15
                )
                delta_y.shift(LEFT*delta_shift) if tip == 1 else delta_y.shift(RIGHT*delta_shift)
                delta_y_label = MathTex(r'\Delta y').scale(label_scale).next_to(delta_y, LEFT) if tip == 1 else MathTex(r'\Delta y').scale(label_scale).next_to(delta_y, RIGHT)
                delta_ys.add(delta_y)
                y_labels.add(delta_y_label)
        
            return delta_xs, x_labels, delta_ys, y_labels
        

        self.wait()

        
        # First square
        delta_xs1, x_labels1, delta_ys1, y_labels1 = create_square_arrows(quad1)
        self.play(Write(field),Create(quad1), GrowArrow(delta_xs1[0]), GrowArrow(delta_xs1[1]), GrowArrow(delta_ys1[0]), GrowArrow(delta_ys1[1]), Write(label1))
        self.wait()

        for i in range(len(delta_xs1)):
            self.play(delta_xs1[i].animate.shift(DOWN*(0.5))) if i ==0 else self.play(delta_xs1[i].animate.shift(UP*0.5))
        
        for i in range(len(delta_ys1)):
            self.play(delta_ys1[i].animate.shift(LEFT*(0.5))) if i ==1 else self.play(delta_ys1[i].animate.shift(RIGHT*0.5))
        

        square1 = VGroup(quad1,delta_xs1,delta_ys1)
        self.play(square1.animate.shift(LEFT*1))

        #Second Square
        quad2 = Square().next_to(quad1, buff=0)
        delta_xs2, x_labels2, delta_ys2, y_labels2 = create_square_arrows(quad2)
        label2= always_redraw(lambda:MathTex(r'\partial{R_2}').scale(0.7).next_to(quad2,DR,buff= 0.2).shift(LEFT*0.4))

        for i in range(len(delta_xs2)):
            delta_xs2[i].shift(DOWN*(0.5)) if i ==0 else delta_xs2[i].shift(UP*0.5)
        
        for i in range(len(delta_ys2)):
            delta_ys2[i].shift(LEFT*(0.5)) if i ==1 else delta_ys2[i].shift(RIGHT*0.5)
        
        square2 = VGroup(quad2,delta_xs2,delta_ys2)

        all = VGroup(square1,square2,field)



        self.play(Create(square2), Write(label2))
        self.wait()

        self.play(all.animate.scale(0.8).to_edge(RIGHT))

        vertices = quad1.get_vertices()
        sides = [
            Line(vertices[i], vertices[(i+1) % 4], color=WHITE)
            for i in range(4)
        ]

        eq1 = MathTex(r'\oint_{\partial {R_1} \cup \partial{R_2}} \vec{F} \cdot d \vec{r}=').scale(0.5).to_edge(LEFT).shift(UP*2)
        eq2 = MathTex(r'\oint_{\partial{R_1}} \vec{F} \cdot d \vec{r} + \oint_{\partial{R_2}} \vec{F} \cdot d \vec{r}').scale(0.5).next_to(eq1,RIGHT)
        
        eq3  = MathTex(r'\oint_{\partial {R_1} \cup \partial{R_2}} \vec{F} \cdot d \vec{r}=').scale(0.5).next_to(eq1,DOWN)
        
        eq4 = MathTex(r'\int_x^{x+\Delta x }P(t,y) dt +\int_y^{y+\Delta y}Q(x +\Delta x,t)dt -').scale(0.5).next_to(eq3,RIGHT)
        eq5 = MathTex(r'\int_x^{x+\Delta x }P(t,y+\Delta y) - \int_y^{y+\Delta y}Q(x ,t)dt +').scale(0.5).next_to(eq4,DOWN)
        
        eq6 = MathTex(r'\int_{x+\Delta x}^{x+2\Delta x }P(t,y) dt +\int_y^{y+\Delta y}Q(x +2\Delta x,t)dt  -').scale(0.5).next_to(eq5,DOWN)
        eq7 = MathTex(r'\int_{x+\Delta x}^{x+2\Delta x }P(t,y +\Delta y) dt -\int_y^{y+\Delta y}Q(x + \Delta x ,t)dt').scale(0.5).next_to(eq6,DOWN).shift(LEFT*0.2)
        
        separatore = DashedLine(UP*3.5 + RIGHT*0.5,DOWN *3.5+ RIGHT*0.5).shift(RIGHT*0.3)
        
        self.play(Create(separatore))


        cross1 = Line(
            start = eq3.get_corner(DL),
            end= eq3.get_corner(UR)
        ).shift(RIGHT*4.7,DOWN*1)
         
        cross2 = Line(
            start = eq3.get_corner(DL),
            end= eq3.get_corner(UR)
        ).shift(DOWN*2.8,RIGHT*4.7)
        

        high1 = eq7[0][-17:]
        high2= eq5[0][-15:]

        ######
        self.play(Write(eq1))
        self.play(Write(eq2))
        self.wait()

        self.play(Write(eq3))
        self.play(Write(eq4))
        self.play(Write(eq5))
        self.play(Write(eq6))
        self.play(Write(eq7))

        self.wait(2)

        self.play(Indicate(high1,color =MAROON_C), Indicate(high2, color = MAROON_C),Indicate(sides[3], color= MAROON_C))
        self.play(Create(cross1), Create(cross2))
        self.wait()        #####


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

        