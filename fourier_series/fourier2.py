from manim import *
import numpy as np
from manim import config
config.frame_rate = 60

class MultipleRotatingVectors(Scene):
    def construct(self):
        angle = ValueTracker(0)

        def update_angle(mob, dt):
            mob.increment_value(dt * PI/ 3)

        vectors = VGroup()
        labels = VGroup()
        circles = VGroup()

        for i in [-2, -1, 0, 1, 2]:
            base_point = np.array([-2.5 * i, 2.5, 0])  # Spread near the top edge
            vector = always_redraw(lambda i=i, base_point=base_point: Arrow(
                start=base_point,
                end=base_point + 0.8 *np.array([
                    np.cos(angle.get_value() * -i),
                    np.sin(angle.get_value() * -i),
                    0
                ]),
                buff=0,
                stroke_width=4))
            
            vectors.add(vector)
                
            # Create a circle around the vectors
            circle = Circle(radius=1, stroke_color = [MAROON_C, BLUE],stroke_width = 2).move_to(base_point).scale(0.8)
            circles.add(circle)

            label = MathTex(
                r"e^{ %d x i}" % (-i),
                font_size=30
            ).next_to(circle, DOWN, buff=0.2).set_color(MAROON_C)  # Label under each vector
            labels.add(label)

    
        left_vector = always_redraw(lambda: Arrow(
            start = np.array([-7, 2.5 ,0]),
            end = np.array([
                -7 + 1/2* np.cos(angle.get_value() * -3), 
                2.5 + 1/2* np.sin(angle.get_value() * -3), 
                0]),
            buff = 0
        ).set_opacity(0.3))
        
        left_circle = Circle(radius= 1/2, stroke_color = [MAROON_C, BLUE],stroke_width = 2,stroke_opacity = 0.3).move_to(np.array([-7, 2.5 ,0]))

        left_label = MathTex(
            r"e^{ %d x i}" % (-3),
            font_size=20
        ).next_to(left_circle, DOWN, buff=0.2).set_color(MAROON_C).set_opacity(0.3)  # Label under each vector

        right_vector = always_redraw(lambda: Arrow(
            start = np.array([7, 2.5 ,0]),
            end = np.array([
                7 + 1/2* np.cos(angle.get_value() * 3), 
                2.5 + 1/2* np.sin(angle.get_value() * 3), 
                0]),
            buff = 0
        ).set_opacity(0.3))
        
        right_circle = Circle(radius= 1/2, stroke_color = [MAROON_C, BLUE],stroke_width = 2, stroke_opacity = 0.3).move_to(np.array([7, 2.5 ,0]))

        right_label = MathTex(
            r"e^{ %d x i}" % (3),
            font_size=20
        ).next_to(right_circle, DOWN, buff=0.2).set_color(MAROON_C).set_opacity(0.3)

        self.add(vectors,circles,labels, right_circle, right_vector, right_label, left_circle, left_label, left_vector)
    
        self.wait()
        angle.add_updater(update_angle)
        self.add(angle)
        self.wait(1)
        
        plane= ComplexPlane(
            x_range = [-2,2,1],
            y_range = [-2,2,1],
            background_line_style = {
                "stroke_color": BLUE,
                "stroke_width": 2,
                "stroke_opacity": 0.5,
            }
        ).to_edge(DOWN, buff = 1).add_coordinates()
        
        real_plane = NumberPlane(
            x_range = [-1,4*PI,PI/2],
            y_range = [-4,4,1],
            y_length= 4,
            x_length = 4,
            axis_config ={
                "include_numbers": False},
            background_line_style = {
                "stroke_color": BLUE,
                "stroke_width": 2,
                "stroke_opacity": 0.5,
            }
        ).to_corner(DR,buff = 0.5)

        self.play(Write(plane))

        # 2) Now manually place π‑tick labels along x‑axis
        x_ticks = np.arange(PI /2, 4 * PI + 0.1, PI / 2)
        x_values = []
        for x in x_ticks:
            # decide what text to show
            if x == 0:
                txt = "0"
            elif x == PI:
                txt = r"\pi"
            elif x == 2 * PI:
                txt = r"2\pi"
            elif x == 3 * PI:
                txt = r"3\pi"
            else:
                # for odd multiples of π/2
                k = int(x / (PI / 2))
                if k % 2 and k!= 1:  # 1,3,5,7...
                    txt = rf"\tfrac{{{k}\pi}}{{2}}"
                elif k == 1:
                    txt = rf"\tfrac{{\pi}}{{2}}"
                else:
                    txt = str(int(x / PI)) + r"\pi"
            label = MathTex(txt, font_size=24)
            # position it just below the x‑axis
            label.next_to(real_plane.c2p(x, 0), DOWN, buff=0.1)
            x_values.append(label)
        
        y_ticks = np.arange(-4,5,1)
        for y in y_ticks:
            txt = str(y)
            label = MathTex(txt, font_size=24).next_to(real_plane.c2p(0,y), LEFT, buff=0.1)
            x_values.append(label)

        t = ValueTracker(0)
        vector1 = always_redraw(lambda: Arrow(
            start = plane.c2p(0,0),
            end = plane.c2p(np.cos(t.get_value()), np.sin(t.get_value())),
            buff = 0
        ))
        circle1 = always_redraw(lambda: Circle(
            radius = 1,
            stroke_color = [MAROON_C, BLUE],
            stroke_width = 2,
            stroke_opacity = 0.3
        ).move_to(vector1.get_start()))


        vector2 = always_redraw(lambda: Arrow(
            start = vector1.get_end(),
            end = plane.c2p(np.cos(t.get_value()) + np.cos(-t.get_value()), np.sin(t.get_value()) + np.sin(-t.get_value())),
            buff = 0
        ))

        circle2 = always_redraw(lambda: Circle(
            radius = 1,
            stroke_color = [MAROON_C, BLUE],
            stroke_width = 2,
            stroke_opacity = 0.3
        ).move_to(vector2.get_start()))
        
        def sum_point():
            x = t.get_value()
            val = np.exp(1j * x) + np.exp(-1j * x)  # complex sum = 2 cos(x)
            return plane.c2p(val.real, val.imag)
    

        func = always_redraw(lambda: 
                             real_plane.plot(lambda x: 2*np.cos(x),
                                             x_range=[0,t.get_value()], color=MAROON_C, stroke_width=2))
        
        func_dot = always_redraw(lambda: Dot().move_to(
            real_plane.c2p(
                t.get_value(), 
                2*np.cos(t.get_value())
                )).set_color(MAROON_C
                             ))

        #self.add(func, func_dot)

        add_to_cos = VGroup(vectors[1], vectors[3], circles[1], circles[3])
        vc12 = VGroup(vector1, vector2, circle1, circle2)

        self.wait(2)
        self.play(TransformFromCopy(add_to_cos, vc12), run_time=3)
        
        #self.add(trace_dot, traced_path)
        
        # 7) Animate the rotation—path is drawn and simultaneously scrolls
        self.play(plane.animate.to_corner(DL, buff=0.5), run_time=2)
        self.wait()
        self.play(
            *[Write(label) for label in x_values],
            Write(real_plane),
            run_time=2,
            rate_func=linear)
        
        trace_dot = always_redraw(lambda: Dot(
            sum_point(),  # invisible
        ).set_color(BLUE))
        

        # 5) A continuous traced path of that dot
        traced_path = TracedPath(
            trace_dot.get_center,
            stroke_color=BLUE,
            stroke_width=3,
        )
        
        sum_exp_label = MathTex('e^{ix} + e^{-ix} = 2\cos(x)').next_to(plane,RIGHT, buff = 0.4).scale(0.8)

        sum_exp_label[0][:8].set_color(BLUE)
        sum_exp_label[0][9:].set_color(MAROON_C)


        self.play(Write(trace_dot),Write(func), Write(func_dot), run_time=1, rate_func=linear)
        self.wait(2)
        self.play(Create(traced_path), run_time=2, rate_func=linear)
        self.play(t.animate.increment_value(4* PI), run_time=12  , rate_func= linear)
        self.play(Write(sum_exp_label))
        self.wait(2)

        self.play(
        *[Uncreate(mob) for mob in self.mobjects if isinstance(mob, VMobject)],
        run_time=3.2
    )
        self.wait(2)

class Finale(Scene):
    def construct(self):

        n = ValueTracker(3.5)
        
        real_axes = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 2,
                "stroke_opacity": 0.2,
            }
        ).add_coordinates().to_edge(RIGHT).shift(DOWN *0.5).scale(0.8)

        funct = always_redraw(lambda: ParametricFunction(
            lambda t: real_axes.c2p(
                      4*np.cos(n.get_value()*t/2)**2 *np.cos(t),
                      4*np.cos(n.get_value()*t/2)**2 *np.sin(t)),
                      t_range = [0, 10 *PI], stroke_width = 2).set_color(GREEN))

        complex_plane = ComplexPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 2,
                "stroke_opacity": 0.2,
            }
        ).add_coordinates().to_edge(LEFT).shift(DOWN *0.5).scale(0.8)

        self.play(Write(complex_plane))

        

        # Tempo
        time = ValueTracker(0)
        self.add(time)

        # Coefficienti e velocità dinamiche
        coeff_trackers = [ValueTracker(t) for t in [2,1,1]]
        vel_trackers = [ValueTracker(v) for v in [1,-3,5]]
        for vt in vel_trackers:
            self.add(vt)

        def get_vectors():
            origin = 0 + 0j
            arrows = []
            for i, v in zip(coeff_trackers, vel_trackers):
                angle = v.get_value() * time.get_value()
                tip = origin + i.get_value() * np.exp(1j * angle)
                arrow = Arrow(
                    start=complex_plane.c2p(origin.real, origin.imag),
                    end=complex_plane.c2p(tip.real, tip.imag),
                    buff=0,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.10,
                    color=WHITE,
                )
                arrows.append(arrow)
                origin = tip
            return VGroup(*arrows)

        def get_circles():
            origin = 0 + 0j
            circles = []
            for i, v in zip(coeff_trackers, vel_trackers):
                circle = Circle(
                    radius=i.get_value(),
                    color=BLUE,
                    stroke_opacity=0.3,
                    stroke_width=1
                ).move_to(complex_plane.c2p(origin.real, origin.imag))
                angle = v.get_value() * time.get_value()
                origin += i.get_value() * np.exp(1j * angle)
                circles.append(circle)
            return VGroup(*circles)

        def get_tip():
            origin = 0 + 0j
            for i, v in zip(coeff_trackers, vel_trackers):
                angle = v.get_value() * time.get_value()
                origin += i.get_value() * np.exp(1j * angle)
            return complex_plane.c2p(origin.real, origin.imag)

        vectors = always_redraw(get_vectors)
        circles = always_redraw(get_circles)

        vect_sum_label  = always_redraw(lambda: MathTex(
            r"ae^{i\omega_1t}+be^{i\omega_2t}+ce^{i\omega_3t}", 
            font_size = 36).next_to(complex_plane,UP,buff = 0.6))
        
        
        cos_formula_label = MathTex(
            r"r(t) = 4b\cos^2\left(\frac{n\theta}{2}\right)", 
            font_size = 36).next_to(real_axes,UP,buff = 0.3)


        
        tip_dot = always_redraw(lambda: Dot(get_tip(), radius=0, color=YELLOW))
        self.add(tip_dot)

        # Trace the dot instead of the raw function
        trace1 = TracedPath(tip_dot.get_center, stroke_color=YELLOW, stroke_width=2)
        
        
        self.play(Create(circles), Create(vectors), Write(vect_sum_label))
        self.wait(0.01)
        self.add(trace1)

        # Prima animazione
        self.play(time.animate.set_value(2 * PI/vel_trackers[0].get_value()), run_time=6, rate_func=smooth)

        self.wait()
        self.play(Write(real_axes), run_time = 2)

        
        self.play(Write(funct), Write(cos_formula_label),run_time = 4)
        
        
        self.wait()
        self.play(n.animate.set_value(4), run_time = 4)
        self.wait()

        
        self.play(Uncreate(trace1), run_time=2)
        

        time.set_value(0)
        self.wait()

        
        vel_trackers[0].set_value(1)
        vel_trackers[1].set_value(-4)
        vel_trackers[2].set_value(6)
        

        self.play(coeff_trackers[0].animate.set_value(2),
                  coeff_trackers[1].animate.set_value(1),
                  coeff_trackers[2].animate.set_value(1))

        
       
        trace2 = TracedPath(tip_dot.get_center, stroke_color=RED, stroke_width=2)
        self.add(trace2)

        
        self.play(time.animate.set_value(2 * PI/vel_trackers[0].get_value() ), n.animate.set_value(5), run_time=6, rate_func=smooth)

        self.play(Uncreate(trace2), run_time = 2)

        time.set_value(0)
        
        vel_trackers[0].set_value(1)
        vel_trackers[1].set_value(-0.4)
        vel_trackers[2].set_value(2.4)
        
        self.play(coeff_trackers[0].animate.set_value(2),
                  coeff_trackers[1].animate.set_value(1),
                  coeff_trackers[2].animate.set_value(1))
 
       
        time.set_value(0)
        self.wait()
        
        trace3 = TracedPath(tip_dot.get_center, stroke_color=BLUE, stroke_width=2)
        self.add(trace3)

        
        self.play(time.animate.set_value(10* PI), n.animate.set_value(1.4), run_time=10, rate_func=smooth)
        
        self.play(Uncreate(trace3), run_time = 2)
        self.wait(2)

     
        self.wait()
        self.play(
                  FadeOut(real_axes),
                  FadeOut(funct),
                  FadeOut(cos_formula_label),
                  run_time = 3)
        self.remove(funct)
        self.wait()
        
        
        self.play(complex_plane.animate.move_to(ORIGIN))
        self.wait(2)
        
class Finale2(Scene):
    def construct(self):
        
        
        complex_plane = ComplexPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 2,
                "stroke_opacity": 0.2,
            }
        ).add_coordinates().scale(0.8).move_to(ORIGIN)

        vect_sum_label  = always_redraw(lambda: MathTex(
            r"ae^{i\omega_1t}+be^{i\omega_2t}+ce^{i\omega_3t}", 
            font_size = 36).next_to(complex_plane,UP,buff = 0.6))
        
        
        self.add(complex_plane,vect_sum_label)
        
        time = ValueTracker(0)
        self.add(time)

        # Coefficienti e velocità dinamiche
        coeff_trackers = [ValueTracker(t) for t in [2,1,1,0,0,0,0,0]]
        vel_trackers = [ValueTracker(v) for v in [1,-3,5,0,0,0,0,0]]
        for vt in vel_trackers:
            self.add(vt)

        def get_vectors():
            origin = 0 + 0j
            arrows = []
            for i, v in zip(coeff_trackers, vel_trackers):
                angle = v.get_value() * time.get_value()
                tip = origin + i.get_value() * np.exp(1j * angle)
                arrow = Arrow(
                    start=complex_plane.c2p(origin.real, origin.imag),
                    end=complex_plane.c2p(tip.real, tip.imag),
                    buff=0,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.10,
                    color=WHITE,
                )
                arrows.append(arrow)
                origin = tip
            return VGroup(*arrows)

        def get_circles():
            origin = 0 + 0j
            circles = []
            for i, v in zip(coeff_trackers, vel_trackers):
                circle = Circle(
                    radius=i.get_value(),
                    color=BLUE,
                    stroke_opacity=0.3,
                    stroke_width=1
                ).move_to(complex_plane.c2p(origin.real, origin.imag))
                angle = v.get_value() * time.get_value()
                origin += i.get_value() * np.exp(1j * angle)
                circles.append(circle)
            return VGroup(*circles)

        def get_tip():
            origin = 0 + 0j
            for i, v in zip(coeff_trackers, vel_trackers):
                angle = v.get_value() * time.get_value()
                origin += i.get_value() * np.exp(1j * angle)
            return complex_plane.c2p(origin.real, origin.imag)

        vectors = always_redraw(get_vectors)
        circles = always_redraw(get_circles)

        self.add(circles, vectors)

        tip_dot = always_redraw(lambda: Dot(get_tip(), radius=0, color=YELLOW))
        self.add(tip_dot)


        vel_trackers[0].set_value(-0.159)
        vel_trackers[1].set_value(0.159)
        vel_trackers[2].set_value(-0.318)
        vel_trackers[3].set_value(0.318)
        vel_trackers[4].set_value(-0.477)
        vel_trackers[5].set_value(0.477)
        vel_trackers[6].set_value(-0.636)
        vel_trackers[7].set_value(0.636)
        
        self.wait()
        self.play(complex_plane.animate.shift(DOWN*1))
        self.wait()

        new_vect_sum = always_redraw(lambda: MathTex(
            r"\sum_{N=-n}^{n}c_ne^{int} ", 
            font_size = 36).next_to(complex_plane,UP,buff = 0.6))
        
        
        self.play(coeff_trackers[0].animate.set_value(2.5),
                  coeff_trackers[1].animate.set_value(0.1),
                  coeff_trackers[2].animate.set_value(-0.5),
                  coeff_trackers[3].animate.set_value(-0.5),
                  coeff_trackers[4].animate.set_value(-0.6),
                  coeff_trackers[5].animate.set_value(0.2),
                  coeff_trackers[6].animate.set_value(0.1),
                  coeff_trackers[7].animate.set_value(-0.1),
                  ReplacementTransform(vect_sum_label,new_vect_sum))
        
        self.wait()


        # Trace the dot instead of the raw function
        trace1 = TracedPath(tip_dot.get_center, stroke_color=YELLOW, stroke_width=2)
        
        
        self.wait()
        

        self.wait(0.01)
        self.add(trace1)

        # Prima animazione
        self.play(time.animate.set_value(12.57* PI), run_time=6, rate_func=smooth)
        self.wait(2)

        
        

        


